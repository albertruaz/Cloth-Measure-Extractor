#!/usr/bin/env python3
"""
실제 쿼리가 잘 작동하는지 테스트하는 스크립트
"""
import pymysql
import yaml
import json

def test_query():
    """실제 쿼리 테스트"""
    with open('db_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    db_config = config['database']
    query_config = config['query']
    
    print("=" * 80)
    print("실제 쿼리 테스트")
    print("=" * 80)
    
    try:
        conn = pymysql.connect(
            host=db_config['host'],
            port=db_config['port'],
            user=db_config['user'],
            password=db_config['password'],
            database=db_config['name'],
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        
        cursor = conn.cursor()
        
        # 실제 사용할 쿼리
        query = """
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
              AND request_uri LIKE '%/measurement'
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
        WHERE member_id IN ({member_ids})
          AND product.created_at >= %s
          AND state = 'PUBLISHED'
          AND category IN ({categories})
        LIMIT 5
        """
        
        # 파라미터 생성
        categories = query_config.get('categories', [])
        member_ids = query_config.get('member_ids', [2, 3])
        date_from = query_config.get('date_from', '2025-10-10')
        
        # 플레이스홀더 생성
        member_ids_ph = ', '.join(['%s'] * len(member_ids))
        categories_ph = ', '.join(['%s'] * len(categories))
        
        query = query.format(member_ids=member_ids_ph, categories=categories_ph)
        params = member_ids + [date_from] + categories
        
        print(f"카테고리: {categories}")
        print(f"Member IDs: {member_ids}")
        print(f"날짜 기준: {date_from}")
        print()
        print("쿼리 실행 중...")
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        print(f"\n✓ 결과: {len(results)}개")
        print("=" * 80)
        
        if results:
            for i, row in enumerate(results, 1):
                print(f"\n[샘플 {i}]")
                print(f"생성일: {row['p_created_at']}")
                print(f"카테고리: {row['category']}")
                print(f"상품명: {row['name']}")
                print(f"이미지 URL: {row['image_uri'][:80]}...")
                print(f"Member ID: {row['member_id']}")
                
                # request_body 파싱
                if row['request_body']:
                    try:
                        body = json.loads(row['request_body'])
                        measurements = body.get('measurements', {})
                        print(f"측정 항목: {list(measurements.keys())}")
                    except:
                        print(f"request_body 파싱 실패")
                
                print("-" * 80)
        else:
            print("\n⚠️  조건에 맞는 데이터가 없습니다.")
            print("\n확인사항:")
            print(f"1. {date_from} 이후에 생성된 상품이 있는지")
            print(f"2. member_id가 {member_ids}인 상품이 있는지")
            print(f"3. 카테고리가 {categories}에 포함되는지")
        
        cursor.close()
        conn.close()
        
        print("\n✓ 테스트 완료!")
        
    except Exception as e:
        print(f"\n✗ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_query()
