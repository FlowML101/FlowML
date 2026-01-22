"""
Quick API test script
Run with: python test_api.py
"""
import requests
import time
import sys

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("Testing /health...")
    r = requests.get(f"{BASE_URL}/health")
    print(f"  Status: {r.status_code}")
    print(f"  Response: {r.json()}")
    return r.status_code == 200

def test_stats():
    """Test stats endpoint"""
    print("\nTesting /api/stats...")
    r = requests.get(f"{BASE_URL}/api/stats")
    print(f"  Status: {r.status_code}")
    print(f"  Response: {r.json()}")
    return r.status_code == 200

def test_workers():
    """Test workers endpoint"""
    print("\nTesting /api/workers...")
    r = requests.get(f"{BASE_URL}/api/workers")
    print(f"  Status: {r.status_code}")
    data = r.json()
    print(f"  Workers found: {len(data)}")
    if data:
        print(f"  Master: {data[0]['hostname']} - CPU: {data[0]['cpu_percent']}%")
    return r.status_code == 200

def test_resources():
    """Test resources endpoint"""
    print("\nTesting /api/stats/resources...")
    r = requests.get(f"{BASE_URL}/api/stats/resources")
    print(f"  Status: {r.status_code}")
    data = r.json()
    print(f"  CPU: {data['cpu_percent']}%")
    print(f"  RAM: {data['ram_used_gb']}/{data['ram_total_gb']} GB ({data['ram_percent']}%)")
    if data.get('vram_percent'):
        print(f"  VRAM: {data['vram_used_gb']}/{data['vram_total_gb']} GB ({data['vram_percent']}%)")
    return r.status_code == 200

def test_datasets():
    """Test datasets endpoint"""
    print("\nTesting /api/datasets...")
    r = requests.get(f"{BASE_URL}/api/datasets")
    print(f"  Status: {r.status_code}")
    print(f"  Datasets: {len(r.json())}")
    return r.status_code == 200

def test_upload():
    """Test dataset upload with sample CSV"""
    print("\nTesting /api/datasets/upload...")
    
    # Create sample CSV
    csv_content = """PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
1,0,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S
2,1,1,"Cumings, Mrs. John Bradley",female,38,1,0,PC 17599,71.2833,C85,C
3,1,3,"Heikkinen, Miss. Laina",female,26,0,0,STON/O2. 3101282,7.925,,S
4,1,1,"Futrelle, Mrs. Jacques Heath",female,35,1,0,113803,53.1,C123,S
5,0,3,"Allen, Mr. William Henry",male,35,0,0,373450,8.05,,S
6,0,3,"Moran, Mr. James",male,,0,0,330877,8.4583,,Q
7,0,1,"McCarthy, Mr. Timothy J",male,54,0,0,17463,51.8625,E46,S
8,0,3,"Palsson, Master. Gosta Leonard",male,2,3,1,349909,21.075,,S
9,1,3,"Johnson, Mrs. Oscar W",female,27,0,2,347742,11.1333,,S
10,1,2,"Nasser, Mrs. Nicholas",female,14,1,0,237736,30.0708,,C"""
    
    files = {'file': ('titanic_test.csv', csv_content, 'text/csv')}
    data = {'name': 'Titanic Test', 'description': 'Test dataset'}
    
    r = requests.post(f"{BASE_URL}/api/datasets/upload", files=files, data=data)
    print(f"  Status: {r.status_code}")
    if r.status_code == 200:
        dataset = r.json()
        print(f"  Dataset ID: {dataset['id']}")
        print(f"  Rows: {dataset['num_rows']}, Columns: {dataset['num_columns']}")
        return dataset['id']
    else:
        print(f"  Error: {r.text}")
    return None

def test_preview(dataset_id):
    """Test dataset preview"""
    print(f"\nTesting /api/datasets/{dataset_id}/preview...")
    r = requests.get(f"{BASE_URL}/api/datasets/{dataset_id}/preview?rows=5")
    print(f"  Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"  Columns: {data['columns']}")
        print(f"  Preview rows: {len(data['preview_rows'])}")
    return r.status_code == 200

def main():
    print("=" * 50)
    print("FlowML API Test Suite")
    print("=" * 50)
    
    results = []
    
    # Basic tests
    results.append(("Health", test_health()))
    results.append(("Stats", test_stats()))
    results.append(("Workers", test_workers()))
    results.append(("Resources", test_resources()))
    results.append(("Datasets List", test_datasets()))
    
    # Upload test
    dataset_id = test_upload()
    results.append(("Upload", dataset_id is not None))
    
    if dataset_id:
        results.append(("Preview", test_preview(dataset_id)))
    
    # Summary
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {name}: {status}")
    
    print(f"\n  Total: {passed}/{total} passed")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Cannot connect to server at http://localhost:8000")
        print("   Make sure the backend is running: python main.py")
        sys.exit(1)
