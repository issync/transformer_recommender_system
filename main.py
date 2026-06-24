import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src import init_db, train, get_recommendations, record_action
from src.database import load_all_products, load_all_users


def main():
    init_db()

    products = load_all_products()
    users = load_all_users()
    num_items = len(products)

    print(f"商品数: {num_items}  用户数: {len(users)}")
    print("开始训练...\n")

    model = train(num_items=num_items, epochs=150, lr=1e-3, batch_size=16)

    print("\n--- 个性化推荐 ---")
    for user in users[:5]:
        uid = user["user_id"]
        recs = get_recommendations(model, uid, top_k=3)
        names = [r["name"] for r in recs]
        print(f"  {user['label']:<8} -> {names}")

    print("\n--- 搜索推荐 ---")
    for kw in ["耳机", "跑步", "编程", "厨房"]:
        recs = get_recommendations(model, user_id=1, keyword=kw, top_k=3)
        names = [r["name"] for r in recs]
        print(f"  搜索「{kw}」-> {names}")

    print("\n--- 实时行为更新 ---")
    record_action(user_id=1, product_id=27, action="purchase")
    recs = get_recommendations(model, user_id=1, top_k=3)
    print(f"  购买《深度学习》后推荐 -> {[r['name'] for r in recs]}")


if __name__ == "__main__":
    main()
