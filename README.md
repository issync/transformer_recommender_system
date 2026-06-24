# Transformer recommender system

基于 Transformer（SASRec）架构的电商推荐系统，支持个性化推荐与关键词搜索融合排序。

## 特性

- **SASRec 序列推荐**：用 Self-Attention 建模用户行为序列，捕捉购买顺序中的兴趣变化
- **实时行为更新**：用户每次点击/购买后，下次推荐立即反映最新兴趣，无需重新训练
- **搜索 + 推荐融合**：关键词搜索过滤候选池，Transformer 对结果重排序
- **冷启动兜底**：新用户无行为时自动按商品评分推荐
- **SQLite + CSV**：数据存储轻量，商品和用户数据通过 CSV 管理

## 项目结构

```
ecom-rec/
├── main.py              # 入口，训练 + 演示
├── requirements.txt
├── LICENSE
├── data/
│   ├── products.csv     # 商品数据（45件）
│   ├── users.csv        # 用户数据（20人）
│   └── behaviors.csv    # 行为数据（320条）
└── src/
    ├── database.py      # 数据库读写
    ├── model.py         # SASRec Transformer 模型
    ├── trainer.py       # 训练逻辑
    └── recommender.py   # 推荐 + 搜索引擎
```

## 快速开始

```bash
# 安装依赖
pip install torch "numpy<2"

# 运行
python main.py
```

## 核心接口

```python
from src import init_db, train, get_recommendations, record_action
from src.database import load_all_products

# 初始化数据库（首次运行）
init_db()

# 训练模型
model = train(num_items=45, epochs=150)

# 个性化推荐（keyword 为空）
recs = get_recommendations(model, user_id=1)

# 搜索 + 推荐排序
recs = get_recommendations(model, user_id=1, keyword="耳机")

# 记录用户行为（实时更新，下次推荐立即生效）
record_action(user_id=1, product_id=3, action="purchase")
# action: "click" | "favorite" | "cart" | "purchase"
```

## 模型架构

```
用户行为序列 [item1, item2, ..., itemN]
        ↓
  商品嵌入 + 位置编码
        ↓
  Transformer Encoder
  (Multi-Head Self-Attention + FFN + LayerNorm) × N层
        ↓
  最后位置向量 = 用户兴趣表示
        ↓
  与所有商品嵌入做点积 → 推荐分数
```

关键设计：
- **因果掩码**：每个位置只能 attend 到自己和之前的商品，符合序列推荐的时序逻辑
- **右侧 Padding**：真实序列左对齐，避免 LayerNorm 在全零输入上产生数值问题
- **梯度裁剪**：`max_norm=1.0`，防止训练不稳定

## 扩展数据

修改 `data/products.csv` 增加商品，修改 `data/behaviors.csv` 增加用户行为，删除 `data/shop.db` 后重新运行即可加载新数据。

积累足够真实用户行为后，重新运行 `python main.py` 训练新模型。

## License

[Apache License 2.0](LICENSE)
