"""
CSV 데이터 전처리 유틸리티
팬츠 카테고리 데이터를 필터링하고 파싱하여 학습 가능한 형태로 변환
"""

import pandas as pd
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    """CSV 데이터를 전처리하는 클래스"""
    
    # 필터링할 팬츠 카테고리
    TARGET_CATEGORIES = ['데님 팬츠', '코튼 팬츠', '트레이닝 팬츠', '슬랙스', '숏 팬츠']
    
    def __init__(self, csv_path: str, output_dir: str):
        """
        Args:
            csv_path: 원본 CSV 파일 경로
            output_dir: 전처리된 데이터를 저장할 디렉토리
        """
        self.csv_path = csv_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def parse_request_body(self, request_body_str: str) -> List[Dict]:
        """
        request_body JSON 문자열을 파싱하여 측정 항목 리스트 반환
        
        Args:
            request_body_str: JSON 형식의 request_body 문자열
            
        Returns:
            측정 항목 리스트 [{'name': 'TOTAL_LENGTH', 'x1': 923, 'y1': 988, 'x2': 962, 'y2': 2134, ...}, ...]
        """
        try:
            data = json.loads(request_body_str)
            return data.get('items', [])
        except (json.JSONDecodeError, KeyError) as e:
            print(f"JSON 파싱 에러: {e}")
            return []
    
    def filter_valid_measurements(self, items: List[Dict]) -> Dict[str, Dict]:
        """
        유효한 측정 항목만 필터링 (좌표가 0이 아닌 것만)
        
        Args:
            items: 측정 항목 리스트
            
        Returns:
            {measurement_name: {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}, ...}
        """
        valid_measurements = {}
        
        for item in items:
            name = item.get('name', '')
            x1, y1, x2, y2 = item.get('x1', 0), item.get('y1', 0), item.get('x2', 0), item.get('y2', 0)
            
            # 좌표가 모두 0이 아닌 경우만 유효
            if x1 != 0 or y1 != 0 or x2 != 0 or y2 != 0:
                valid_measurements[name] = {
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'value': item.get('value', 0),
                    'unit': item.get('unit', 'CM')
                }
        
        return valid_measurements
    
    def process_csv(self) -> pd.DataFrame:
        """
        CSV 파일을 읽고 팬츠 카테고리만 필터링하여 전처리
        
        Returns:
            전처리된 DataFrame
        """
        print(f"CSV 파일 읽는 중: {self.csv_path}")
        
        # CSV 읽기 (탭으로 구분)
        df = pd.read_csv(self.csv_path, sep='\t')
        
        print(f"전체 데이터 개수: {len(df)}")
        print(f"카테고리 분포:\n{df['category'].value_counts()}")
        
        # 팬츠 카테고리만 필터링
        df_filtered = df[df['category'].isin(self.TARGET_CATEGORIES)].copy()
        
        print(f"\n필터링된 팬츠 데이터 개수: {len(df_filtered)}")
        print(f"필터링된 카테고리 분포:\n{df_filtered['category'].value_counts()}")
        
        # 필요한 컬럼만 선택하고 ID 생성
        df_filtered['id'] = df_filtered['image_uri']
        
        # request_body 파싱
        print("\nrequest_body 파싱 중...")
        measurements_list = []
        
        for idx, row in df_filtered.iterrows():
            request_body = row['request_body']
            items = self.parse_request_body(request_body)
            valid_measurements = self.filter_valid_measurements(items)
            measurements_list.append(valid_measurements)
        
        df_filtered['measurements'] = measurements_list
        
        # 측정 항목이 없는 데이터 제거
        df_filtered = df_filtered[df_filtered['measurements'].apply(lambda x: len(x) > 0)]
        
        print(f"유효한 측정 항목이 있는 데이터 개수: {len(df_filtered)}")
        
        # 최종 컬럼 선택
        df_final = df_filtered[['id', 'image_uri', 'category', 'measurements']].copy()
        
        return df_final
    
    def split_dataset_by_category(self, df: pd.DataFrame, train_ratio: float = 0.8, val_ratio: float = 0.1, random_seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        카테고리별로 데이터셋을 train/val/test로 분할 (8:1:1)
        
        Args:
            df: 전처리된 DataFrame
            train_ratio: Train 비율 (기본 0.8)
            val_ratio: Validation 비율 (기본 0.1)
            random_seed: 랜덤 시드
            
        Returns:
            (train_df, val_df, test_df)
        """
        train_dfs = []
        val_dfs = []
        test_dfs = []
        
        # 카테고리별로 분할
        for category in df['category'].unique():
            category_df = df[df['category'] == category].copy()
            
            # 먼저 train과 temp로 분할 (train: 80%, temp: 20%)
            train_df, temp_df = train_test_split(
                category_df,
                test_size=(1 - train_ratio),
                random_state=random_seed
            )
            
            # temp를 val과 test로 분할 (val: 10%, test: 10%)
            val_df, test_df = train_test_split(
                temp_df,
                test_size=0.5,  # temp의 절반씩
                random_state=random_seed
            )
            
            train_dfs.append(train_df)
            val_dfs.append(val_df)
            test_dfs.append(test_df)
            
            print(f"\n{category}:")
            print(f"  Train: {len(train_df)} 샘플")
            print(f"  Val: {len(val_df)} 샘플")
            print(f"  Test: {len(test_df)} 샘플")
        
        # 합치기
        train_final = pd.concat(train_dfs, ignore_index=True)
        val_final = pd.concat(val_dfs, ignore_index=True)
        test_final = pd.concat(test_dfs, ignore_index=True)
        
        print(f"\n전체 분할 결과:")
        print(f"  Train: {len(train_final)} 샘플")
        print(f"  Val: {len(val_final)} 샘플")
        print(f"  Test: {len(test_final)} 샘플")
        
        return train_final, val_final, test_final
    
    def save_processed_data(self, df: pd.DataFrame, train_df: pd.DataFrame = None, val_df: pd.DataFrame = None, test_df: pd.DataFrame = None):
        """
        전처리된 데이터를 저장
        
        Args:
            df: 전처리된 DataFrame (전체)
            train_df: Train DataFrame (선택사항)
            val_df: Validation DataFrame (선택사항)
            test_df: Test DataFrame (선택사항)
        """
        # CSV로 저장 (measurements는 JSON 문자열로 변환)
        df_to_save = df.copy()
        df_to_save['measurements'] = df_to_save['measurements'].apply(json.dumps)
        
        csv_output_path = self.output_dir / 'processed_pants_data.csv'
        df_to_save.to_csv(csv_output_path, index=False)
        print(f"\n전처리된 CSV 저장: {csv_output_path}")
        
        # JSON으로도 저장 (더 편한 접근을 위해)
        json_output_path = self.output_dir / 'processed_pants_data.json'
        df.to_json(json_output_path, orient='records', indent=2, force_ascii=False)
        print(f"전처리된 JSON 저장: {json_output_path}")
        
        # Train/Val/Test 분할 데이터 저장
        if train_df is not None and val_df is not None and test_df is not None:
            for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
                split_to_save = split_df.copy()
                split_to_save['measurements'] = split_to_save['measurements'].apply(json.dumps)
                split_path = self.output_dir / f'{split_name}.csv'
                split_to_save.to_csv(split_path, index=False)
                print(f"{split_name.capitalize()} CSV 저장: {split_path}")
        
        # 통계 정보 저장
        stats = {
            'total_samples': len(df),
            'categories': df['category'].value_counts().to_dict(),
            'measurement_types': {}
        }
        
        # 각 측정 항목별 개수 계산
        for measurements in df['measurements']:
            for measure_name in measurements.keys():
                stats['measurement_types'][measure_name] = stats['measurement_types'].get(measure_name, 0) + 1
        
        stats_path = self.output_dir / 'data_statistics.json'
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"데이터 통계 저장: {stats_path}")
        
        print("\n=== 데이터 통계 ===")
        print(f"총 샘플 수: {stats['total_samples']}")
        print(f"카테고리별 분포: {stats['categories']}")
        print(f"측정 항목별 개수: {stats['measurement_types']}")
        
        return stats


def main():
    """메인 실행 함수"""
    # 프로젝트 루트 디렉토리
    project_root = Path(__file__).parent.parent.parent
    
    # 입력 CSV 경로
    csv_path = project_root / 'data' / 'csv_data' / 'size_measurement_data_include_image_url.csv'
    
    # 출력 디렉토리
    output_dir = project_root / 'data' / 'processed'
    
    # 전처리 실행
    preprocessor = DataPreprocessor(str(csv_path), str(output_dir))
    df_processed = preprocessor.process_csv()
    
    # 카테고리별로 8:1:1 분할
    print("\n데이터셋 분할 중 (카테고리별 8:1:1)...")
    train_df, val_df, test_df = preprocessor.split_dataset_by_category(df_processed, train_ratio=0.8, val_ratio=0.1, random_seed=42)
    
    # 저장
    stats = preprocessor.save_processed_data(df_processed, train_df, val_df, test_df)
    
    print("\n✅ 데이터 전처리 및 분할 완료!")
    print(f"전처리된 데이터 위치: {output_dir}")


if __name__ == '__main__':
    main()



