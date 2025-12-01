"""Fetch measurement data from a remote MySQL DB (optionally via SSH tunnel)."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager
import logging

import pandas as pd
import pymysql
import yaml
from sshtunnel import SSHTunnelForwarder

logger = logging.getLogger(__name__)


@dataclass
class SSHConfig:
    enabled: bool = False
    ssh_host: str = ""
    ssh_port: int = 22
    ssh_user: str = ""
    ssh_private_key: Optional[str] = None
    remote_bind_host: Optional[str] = None
    remote_bind_port: Optional[int] = None


@dataclass
class DBConfig:
    host: str = "127.0.0.1"
    port: int = 3306
    user: str = ""
    password: str = ""
    name: str = ""
    table: str = ""
    columns: Dict[str, str] = field(
        default_factory=lambda: {
            "image_uri": "image_uri",
            "category": "category",
            "request_body": "request_body",
        }
    )


@dataclass
class QueryConfig:
    limit: Optional[int] = None
    category_whitelist: Optional[List[str]] = None
    order_by: Optional[str] = None
    extra_where: Optional[str] = None


class RemoteDBFetcher:
    """Load measurement rows from MySQL and save as a TSV for preprocessing."""

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        with open(self.config_path, "r", encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f) or {}

        self.base_dir = self.config_path.parent
        self.ssh = self._parse_ssh(raw_cfg.get("ssh_tunnel", {}))
        self.db = self._parse_db(raw_cfg.get("database", {}))
        self.query = self._parse_query(raw_cfg.get("query", {}))
        self.output_path = self._resolve_path(
            raw_cfg.get("output", {}).get(
                "csv_path", "data/csv_data/remote_measurements.csv"
            )
        )

    def _resolve_path(self, path_like: str) -> Path:
        path = Path(path_like)
        if not path.is_absolute():
            return (self.base_dir / path).resolve()
        return path

    def _parse_ssh(self, cfg: Dict) -> SSHConfig:
        return SSHConfig(
            enabled=bool(cfg.get("enabled", False)),
            ssh_host=cfg.get("ssh_host", ""),
            ssh_port=int(cfg.get("ssh_port", 22)),
            ssh_user=cfg.get("ssh_user", ""),
            ssh_private_key=cfg.get("ssh_private_key"),
            remote_bind_host=cfg.get("remote_bind_host"),
            remote_bind_port=cfg.get("remote_bind_port"),
        )

    def _parse_db(self, cfg: Dict) -> DBConfig:
        return DBConfig(
            host=cfg.get("host", "127.0.0.1"),
            port=int(cfg.get("port", 3306)),
            user=cfg.get("user", ""),
            password=cfg.get("password", ""),
            name=cfg.get("name", ""),
            table=cfg.get("table", ""),
            columns=cfg.get(
                "columns",
                {
                    "image_uri": "image_uri",
                    "category": "category",
                    "request_body": "request_body",
                },
            ),
        )

    def _parse_query(self, cfg: Dict) -> QueryConfig:
        whitelist = cfg.get("category_whitelist")
        return QueryConfig(
            limit=cfg.get("limit"),
            category_whitelist=whitelist if whitelist else None,
            order_by=cfg.get("order_by"),
            extra_where=cfg.get("where"),
        )

    @contextmanager
    def _open_tunnel(self) -> Tuple[str, int]:
        """Open SSH tunnel if enabled and yield (host, port) for DB connection."""
        tunnel = None
        host = self.db.host
        port = self.db.port

        if self.ssh.enabled:
            remote_host = self.ssh.remote_bind_host or self.db.host
            remote_port = self.ssh.remote_bind_port or self.db.port
            pkey = (
                str(Path(self.ssh.ssh_private_key).expanduser())
                if self.ssh.ssh_private_key
                else None
            )
            tunnel = SSHTunnelForwarder(
                (self.ssh.ssh_host, self.ssh.ssh_port),
                ssh_username=self.ssh.ssh_user,
                ssh_pkey=pkey,
                remote_bind_address=(remote_host, remote_port),
            )
            tunnel.start()
            host = "127.0.0.1"
            port = tunnel.local_bind_port
            logger.info(
                "SSH 터널 연결 완료: %s:%s -> %s:%s (local %s)",
                self.ssh.ssh_host,
                self.ssh.ssh_port,
                remote_host,
                remote_port,
                port,
            )

        try:
            yield host, port
        finally:
            if tunnel:
                tunnel.stop()
                logger.info("SSH 터널 종료")

    def _build_query(
        self, categories: Optional[List[str]], limit_override: Optional[int]
    ) -> Tuple[str, List]:
        cols = self.db.columns
        select_cols = [
            f"{cols['image_uri']} AS image_uri",
            f"{cols['category']} AS category",
            f"{cols['request_body']} AS request_body",
        ]
        sql = f"SELECT {', '.join(select_cols)} FROM {self.db.table}"
        clauses: List[str] = []
        params: List = []

        if categories:
            placeholders = ", ".join(["%s"] * len(categories))
            clauses.append(f"{cols['category']} IN ({placeholders})")
            params.extend(categories)

        if self.query.extra_where:
            clauses.append(self.query.extra_where)

        if clauses:
            sql += " WHERE " + " AND ".join(clauses)

        if self.query.order_by:
            sql += f" ORDER BY {self.query.order_by}"

        effective_limit = limit_override or self.query.limit
        if effective_limit:
            sql += " LIMIT %s"
            params.append(int(effective_limit))

        return sql, params

    def fetch_dataframe(
        self,
        categories_override: Optional[List[str]] = None,
        limit_override: Optional[int] = None,
    ) -> pd.DataFrame:
        """Return measurement rows as DataFrame."""
        categories = (
            categories_override
            if categories_override is not None
            else self.query.category_whitelist
        )

        sql, params = self._build_query(categories, limit_override)
        logger.info("DB 쿼리 실행: %s (params=%s)", sql, params)

        with self._open_tunnel() as (host, port):
            conn = pymysql.connect(
                host=host,
                port=port,
                user=self.db.user,
                password=self.db.password,
                db=self.db.name,
                cursorclass=pymysql.cursors.DictCursor,
                charset="utf8mb4",
            )
            try:
                df = pd.read_sql(sql, conn, params=params)
            finally:
                conn.close()
        return df

    def fetch_to_csv(
        self,
        output_csv: Optional[Path] = None,
        categories_override: Optional[List[str]] = None,
        limit_override: Optional[int] = None,
    ) -> Path:
        """Fetch data and persist as TSV for downstream preprocessing."""
        output_path = self._resolve_path(output_csv) if output_csv else self.output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df = self.fetch_dataframe(categories_override, limit_override)
        if df.empty:
            logger.warning("DB에서 가져온 데이터가 없습니다.")
        df.to_csv(output_path, sep="\t", index=False)
        logger.info("DB 데이터 TSV 저장: %s (rows=%d)", output_path, len(df))
        return output_path

