#!/usr/bin/env python3
"""
DB에서 데이터를 가져와 CSV로 저장하는 스크립트

사용법:
    python fetch_data.py
    python fetch_data.py --output data/custom_output.csv
"""
import argparse
import logging
import sys
import os
import json
from pathlib import Path
import pandas as pd
import pymysql
from contextlib import contextmanager
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# .env 파일 로드
load_dotenv()


class DataFetcher:
    """DB에서 측정 데이터를 가져오는 클래스"""
    
    def __init__(self, config_path: str = 'db_config.yaml', target_category: str = None):
        """환경 변수(.env)와 설정 파일(db_config.yaml)에서 로드"""
        # .env에서 DB 연결 정보 로드
        self.ssh_enabled = os.getenv('SSH_ENABLED', 'false').lower() == 'true'
        self.ssh_host = os.getenv('SSH_HOST', '')
        self.ssh_user = os.getenv('SSH_USER', '')
        self.ssh_key_file = os.getenv('SSH_KEY_FILE', '')
        self.ssh_local_port = int(os.getenv('SSH_LOCAL_PORT', 3307))
        
        self.db_host = os.getenv('DB_HOST', 'localhost')
        self.db_port = int(os.getenv('DB_PORT', 3307))
        self.db_user = os.getenv('DB_USER', '')
        self.db_password = os.getenv('DB_PASSWORD', '')
        self.db_name = os.getenv('DB_NAME', '')
        self.db_remote_host = os.getenv('DB_REMOTE_HOST', '')
        self.db_remote_port = int(os.getenv('DB_REMOTE_PORT', 3306))
        
        # 카테고리 프리셋 정의
        self.CATEGORY_PRESETS = {
            "tops": {
                "categories_name": ["tops"],
                "categories": [
                    "맨투맨, 후드", "니트", "블라우스, 셔츠", "반팔 티셔츠", "긴팔 티셔츠",
                    "가디건", "자켓", "코트", "바람막이, 져지", "패딩"
                ]
            },
            "sleeveless_onepiece_vest": {
                "categories_name": ["sleeveless_onepiece_vest"],
                "categories": [
                    "민소매", "미니 원피스", "미디 원피스", "롱 원피스", "점프수트", "베스트"
                ]
            },
            "pants": {
                "categories_name": ["pants"],
                "categories": [
                    "데님 팬츠", "코튼 팬츠", "슬랙스", "숏 팬츠", "트레이닝 팬츠"
                ]
            },
            "skirts": {
                "categories_name": ["skirts"],
                "categories": [
                    "미니 스커트", "미디 스커트", "롱 스커트"
                ]
            }
        }
        
        # db_config.yaml에서 쿼리 설정 로드
        import yaml
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            query_config = config.get('query', {})
            
            # target_category가 지정되면 해당 프리셋 강제 적용
            if target_category and target_category in self.CATEGORY_PRESETS:
                logger.info(f"CLI 옵션 적용: --category {target_category}")
                preset = self.CATEGORY_PRESETS[target_category]
                self.categories_name = preset['categories_name']
                self.categories = preset['categories']
            else:
                self.categories_name = query_config.get('categories_name', [])
                self.categories = query_config.get('categories', [])
                
            self.date_from = query_config.get('date_from', '2025-10-10')
            self.member_ids = query_config.get('member_ids', [2, 3])
            self.limit = query_config.get('limit')
        except FileNotFoundError:
            logger.warning(f"설정 파일 {config_path}를 찾을 수 없습니다. 기본값 사용")
            
            # 파일이 없어도 target_category가 있으면 프리셋 적용
            if target_category and target_category in self.CATEGORY_PRESETS:
                logger.info(f"CLI 옵션 적용: --category {target_category}")
                preset = self.CATEGORY_PRESETS[target_category]
                self.categories_name = preset['categories_name']
                self.categories = preset['categories']
            else:
                self.categories_name = []
                self.categories = []
                
            self.date_from = '2025-10-10'
            self.member_ids = [2, 3]
            self.limit = None
        
        self.tunnel = None
    
    def _setup_ssh_tunnel(self):
        """SSH 터널 설정"""
        if not self.ssh_enabled:
            return None
        
        try:
            from sshtunnel import SSHTunnelForwarder
            
            logger.info(f"SSH 터널 설정 중: {self.ssh_user}@{self.ssh_host}")
            logger.info(f"로컬 포트 {self.ssh_local_port} → {self.db_remote_host}:{self.db_remote_port}")
            
            tunnel = SSHTunnelForwarder(
                (self.ssh_host, 22),
                ssh_username=self.ssh_user,
                ssh_pkey=self.ssh_key_file,
                remote_bind_address=(self.db_remote_host, self.db_remote_port),
                local_bind_address=('localhost', self.ssh_local_port)
            )
            tunnel.start()
            logger.info(f"✓ SSH 터널 연결 성공")
            return tunnel
            
        except ImportError as e:
            logger.error(f"sshtunnel 임포트 에러 상세: {e}")
            logger.error("sshtunnel 패키지가 필요합니다: pip install sshtunnel")
            sys.exit(1)
        except Exception as e:
            logger.error(f"SSH 터널 연결 실패: {e}")
            sys.exit(1)
    
    @contextmanager
    def get_connection(self):
        """DB 연결 (SSH 터널 포함)"""
        # SSH 터널 설정
        if self.ssh_enabled:
            self.tunnel = self._setup_ssh_tunnel()
        
        conn = None
        try:
            conn = pymysql.connect(
                host=self.db_host,
                port=self.db_port,
                user=self.db_user,
                password=self.db_password,
                database=self.db_name,
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor
            )
            yield conn
        except Exception as e:
            logger.error(f"DB 연결 실패: {e}")
            logger.error(f"다음을 확인하세요:")
            logger.error(f"1. SSH 터널이 실행 중인가?")
            logger.error(f"   ssh -i vingle.pem ubuntu@13.125.49.0 -L 3307:vingle-ai-rds-instance-1.c5gyzd5rkihd.ap-northeast-2.rds.amazonaws.com:3306 -N &")
            logger.error(f"2. .env 파일의 DB 설정이 올바른가?")
            raise
        finally:
            if conn:
                conn.close()
            if self.tunnel:
                self.tunnel.stop()
                logger.info("SSH 터널 종료")
    
    def build_query(self):
        """실제 쿼리 생성 (복잡한 JOIN 포함)"""
        # self에 저장된 쿼리 설정 사용
        categories = self.categories
        date_from = self.date_from
        member_ids = self.member_ids
        limit = self.limit
        
        # 실제 사용하는 복잡한 쿼리
        sql = """
        WITH main_image AS (
            SELECT * FROM product_image
            WHERE image_type = 'MAIN'
            ORDER BY id DESC
        ),
        measurement_data AS (
            SELECT
                created_at,
                REGEXP_SUBSTR(request_uri, '[0-9]+') AS product_id,
                request_uri, request_body, request_size, response_status
            FROM request_log
            WHERE method = 'PATCH'
              AND request_uri LIKE '%%/measurement'
            ORDER BY created_at DESC
        ),
        measurement_date_each_1 AS (
            SELECT * FROM (
                SELECT
                    *,
                    ROW_NUMBER() OVER (PARTITION BY product_id ORDER BY created_at DESC) AS rn
                FROM measurement_data
            ) t
            WHERE rn = 1
        )
        SELECT
            product.created_at AS p_created_at,
            image_uri,
            request_uri,
            request_body,
            request_size,
            response_status,
            category,
            name,
            style,
            member_id
        FROM measurement_date_each_1
        INNER JOIN product ON product.id = measurement_date_each_1.product_id
        INNER JOIN main_image ON product.id = main_image.product_id
        WHERE member_id IN (%s)
          AND product.created_at >= %s
          AND state = 'PUBLISHED'
          AND category IN (%s)
        """
        
        params = []
        
        # member_ids 파라미터
        member_ids_placeholder = ', '.join(['%s'] * len(member_ids))
        sql = sql.replace('member_id IN (%s)', f'member_id IN ({member_ids_placeholder})')
        params.extend(member_ids)
        
        # date_from 파라미터
        params.append(date_from)
        
        # categories 파라미터
        categories_placeholder = ', '.join(['%s'] * len(categories))
        sql = sql.replace('category IN (%s)', f'category IN ({categories_placeholder})')
        params.extend(categories)
        
        # LIMIT 추가
        if limit:
            sql += " LIMIT %s"
            params.append(limit)
        
        return sql, params
    
    def fetch(self, output_path: str):
        """데이터 가져와서 CSV로 저장"""
        sql, params = self.build_query()
        logger.info(f"쿼리 실행 중...")
        logger.info(f"파라미터: {params}")
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            results = cursor.fetchall()
            df = pd.DataFrame(results)
        
        logger.info(f"데이터 {len(df)}개 가져옴")
        
        # 출력 디렉토리 생성
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # CSV 저장
        df.to_csv(output_file, index=False, sep='\t')
        logger.info(f"데이터 저장 완료: {output_file}")
        
        return df
    
    def _extract_measurements(self, df):
        """request_body JSON에서 measurement names 추출 + 통계 정보"""
        all_measurements = set()
        measurement_counts = {}
        category_counts = {}
        
        for idx, row in df.iterrows():
            # 카테고리 카운트
            category = row.get('category', 'Unknown')
            category_counts[category] = category_counts.get(category, 0) + 1
            
            # Measurement 카운트
            try:
                body = row['request_body']
                data = json.loads(body)
                
                # 'items' 배열에서 measurement 추출
                if 'items' in data:
                    for item in data['items']:
                        if 'name' in item:
                            name = item['name']
                            all_measurements.add(name)
                            measurement_counts[name] = measurement_counts.get(name, 0) + 1
                # 또는 'measurements' 배열
                elif 'measurements' in data:
                    for measurement in data['measurements']:
                        if 'name' in measurement:
                            name = measurement['name']
                            all_measurements.add(name)
                            measurement_counts[name] = measurement_counts.get(name, 0) + 1
            except Exception as e:
                continue
        
        stats = {
            "measurement_names": sorted(list(all_measurements)),
            "total_samples": len(df),
            "categories": category_counts,
            "measurement_types": measurement_counts
        }
        
        return stats
    
    def _split_data(self, df, train_ratio=0.7, val_ratio=0.15, random_state=42):
        """데이터를 train/val/test로 분할"""
        # train + (val + test)
        train_df, temp_df = train_test_split(
            df, train_size=train_ratio, random_state=random_state, shuffle=True
        )
        
        # val과 test 분할
        val_size = val_ratio / (1 - train_ratio)
        val_df, test_df = train_test_split(
            temp_df, train_size=val_size, random_state=random_state, shuffle=True
        )
        
        return train_df, val_df, test_df
    
    def fetch_by_category(self, output_dir: str = 'data/raw'):
        """카테고리별로 개별 CSV 파일 생성 + train/val/test 분할"""
        raw_dir = Path(output_dir)
        processed_dir = Path('data/processed')
        raw_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # categories_name이 있으면 그걸 파일명으로, 없으면 개별 카테고리명 사용
        if self.categories_name:
            # 여러 카테고리를 하나로 묶어서 저장
            filename = '_'.join(self.categories_name)
            
            logger.info(f"\n{'='*60}")
            logger.info(f"그룹: {filename}")
            logger.info(f"카테고리들: {self.categories}")
            logger.info(f"{'='*60}")
            
            # 모든 카테고리 데이터를 한번에 가져옴
            sql, params = self.build_query()
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, params)
                results = cursor.fetchall()
                df = pd.DataFrame(results)
            
            logger.info(f"✓ 총 {len(df)}개 데이터")
            
            if len(df) > 0:
                # 1. 원본 CSV 저장 (data/raw/)
                raw_file = raw_dir / f"{filename}.csv"
                df.to_csv(raw_file, index=False, sep='\t')
                logger.info(f"  원본 저장: {raw_file}")
                
                # 2. Train/Val/Test 분할
                train_df, val_df, test_df = self._split_data(df)
                
                # 3. 분할된 CSV 저장 (data/processed/)
                train_file = processed_dir / f"{filename}_train.csv"
                val_file = processed_dir / f"{filename}_val.csv"
                test_file = processed_dir / f"{filename}_test.csv"
                
                train_df.to_csv(train_file, index=False, sep='\t')
                val_df.to_csv(val_file, index=False, sep='\t')
                test_df.to_csv(test_file, index=False, sep='\t')
                
                logger.info(f"  Train: {train_file} ({len(train_df)}개)")
                logger.info(f"  Val: {val_file} ({len(val_df)}개)")
                logger.info(f"  Test: {test_file} ({len(test_df)}개)")
                
                # 4. Measurement 통계 추출 (raw, train, val, test 모두)
                stats_raw = self._extract_measurements(df)
                stats_train = self._extract_measurements(train_df)
                stats_val = self._extract_measurements(val_df)
                stats_test = self._extract_measurements(test_df)
                
                # 5. 통합 통계 JSON 생성
                full_stats = {
                    "measurement_names": stats_raw['measurement_names'],
                    "raw": stats_raw,
                    "train": stats_train,
                    "val": stats_val,
                    "test": stats_test
                }
                
                logger.info(f"  Measurements: {full_stats['measurement_names']}")
                logger.info(f"  Raw - Total: {stats_raw['total_samples']}, Categories: {stats_raw['categories']}")
                logger.info(f"  Train - Total: {stats_train['total_samples']}")
                logger.info(f"  Val - Total: {stats_val['total_samples']}")
                logger.info(f"  Test - Total: {stats_test['total_samples']}")
                
                # 6. Measurement stats JSON 저장 (raw 폴더에)
                measurements_file = raw_dir / f"{filename}_measurements.json"
                with open(measurements_file, 'w', encoding='utf-8') as f:
                    json.dump(full_stats, f, indent=2, ensure_ascii=False)
                logger.info(f"  Measurements JSON: {measurements_file}")
        else:
            # 기존 방식: 카테고리별로 개별 저장
            category_mapping = {
                "데님 팬츠": "denim_pants",
                "코튼 팬츠": "cotton_pants", 
                "트레이닝 팬츠": "training_pants",
                "슬랙스": "slacks",
                "숏 팬츠": "short_pants"
            }
            
            for category in self.categories:
                filename = category_mapping.get(category, category)
                
                logger.info(f"\n{'='*60}")
                logger.info(f"카테고리: {category}")
                logger.info(f"{'='*60}")
                
                # 임시로 카테고리 1개만 설정
                original_categories = self.categories
                self.categories = [category]
                
                try:
                    sql, params = self.build_query()
                    
                    with self.get_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute(sql, params)
                        results = cursor.fetchall()
                        df = pd.DataFrame(results)
                    
                    logger.info(f"✓ {category}: {len(df)}개 데이터")
                    
                    if len(df) > 0:
                        # 1. 원본 CSV 저장 (data/raw/)
                        raw_file = raw_dir / f"{filename}.csv"
                        df.to_csv(raw_file, index=False, sep='\t')
                        logger.info(f"  원본 저장: {raw_file}")
                        
                        # 2. Train/Val/Test 분할
                        train_df, val_df, test_df = self._split_data(df)
                        
                        # 3. 분할된 CSV 저장 (data/processed/)
                        train_file = processed_dir / f"{filename}_train.csv"
                        val_file = processed_dir / f"{filename}_val.csv"
                        test_file = processed_dir / f"{filename}_test.csv"
                        
                        train_df.to_csv(train_file, index=False, sep='\t')
                        val_df.to_csv(val_file, index=False, sep='\t')
                        test_df.to_csv(test_file, index=False, sep='\t')
                        
                        logger.info(f"  Train: {train_file} ({len(train_df)}개)")
                        logger.info(f"  Val: {val_file} ({len(val_df)}개)")
                        logger.info(f"  Test: {test_file} ({len(test_df)}개)")
                        
                        # 4. Measurement 통계 추출 (raw, train, val, test 모두)
                        stats_raw = self._extract_measurements(df)
                        stats_train = self._extract_measurements(train_df)
                        stats_val = self._extract_measurements(val_df)
                        stats_test = self._extract_measurements(test_df)
                        
                        # 5. 통합 통계 JSON 생성
                        full_stats = {
                            "measurement_names": stats_raw['measurement_names'],
                            "raw": stats_raw,
                            "train": stats_train,
                            "val": stats_val,
                            "test": stats_test
                        }
                        
                        logger.info(f"  Measurements: {full_stats['measurement_names']}")
                        logger.info(f"  Raw - Total: {stats_raw['total_samples']}, Categories: {stats_raw['categories']}")
                        logger.info(f"  Train - Total: {stats_train['total_samples']}")
                        logger.info(f"  Val - Total: {stats_val['total_samples']}")
                        logger.info(f"  Test - Total: {stats_test['total_samples']}")
                        
                        # 6. Measurement stats JSON 저장 (raw 폴더에)
                        measurements_file = raw_dir / f"{filename}_measurements.json"
                        with open(measurements_file, 'w', encoding='utf-8') as f:
                            json.dump(full_stats, f, indent=2, ensure_ascii=False)
                        logger.info(f"  Measurements JSON: {measurements_file}")
                    else:
                        logger.warning(f"  {category} 데이터 없음")
                        
                finally:
                    # 원래 카테고리 복원
                    self.categories = original_categories


def main():
    parser = argparse.ArgumentParser(description='DB에서 데이터 가져오기')
    parser.add_argument('--output', type=str, default='data/raw_data.csv',
                        help='출력 CSV 파일 경로 (기본: data/raw_data.csv)')
    parser.add_argument('--config', type=str, default='db_config.yaml',
                        help='쿼리 설정 파일 경로 (기본: db_config.yaml)')
    parser.add_argument('--by-category', action='store_true',
                        help='카테고리별로 개별 CSV 파일 생성')
    parser.add_argument('--output-dir', type=str, default='data/raw',
                        help='카테고리별 CSV 저장 디렉토리 (기본: data/raw)')
    parser.add_argument('--category', type=str, 
                        help='데이터를 가져올 카테고리 그룹 (tops, sleeveless_onepiece_vest, pants, skirts)')
    
    args = parser.parse_args()
    
    fetcher = DataFetcher(args.config, target_category=args.category)
    
    # 기본 동작: 카테고리별로 개별 파일 생성
    fetcher.fetch_by_category(args.output_dir if args.by_category else 'data/raw')


if __name__ == '__main__':
    main()
