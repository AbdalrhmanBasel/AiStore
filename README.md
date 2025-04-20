# 🛍️ AiStore — AI-Powered eCommerce with GNN Recommendations

Welcome to **AiStore**, an intelligent eCommerce web application built by **Ai Solutions**. This project integrates a powerful **Graph Neural Network (GNN)** recommendation engine—implemented using **PyTorch**—into a full-featured **Django** web app.

> Personalize user experience using AI models in a real-world online store setting.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Django](https://img.shields.io/badge/Django-4.x-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 🚀 Key Features

- 🧠 **GraphSAGE GNN** for personalized product recommendations  
- 🛒 Full-featured **Django eCommerce** backend (products, users, carts, orders)  
- 🧩 Modular and scalable architecture  
- 📊 Model training, evaluation, and deployment workflows included  
- 🔌 Easy-to-use REST API for recommendations (coming soon)



## 🧠 What Makes This Unique?

Traditional recommendation systems use collaborative filtering or content-based approaches. AiStore goes further by:

- Building a **product-user interaction graph**
- Learning **graph embeddings** with **GraphSAGE**
- Using those embeddings for **real-time recommendations**


## 📁 Project Structure

```bash
aistore/
├── ecommerce/          # Django project settings
├── products/           # Product models and views
├── recommender/        # GNN model training and inference
│   ├── model.py
│   ├── train.py
│   ├── recommend.py
│   └── data_loader.py
├── templates/          # HTML templates
├── static/             # Static files (CSS, JS, Images)
├── manage.py
├── requirements.txt
└── README.md
```
---

## ⚙️ Installation Guide

Follow these steps to run the project locally:

```bash
# 1. Clone the repo
git clone https://github.com/ai-solutions/AiStore.git
cd AiStore

# 2. Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run database migrations
python manage.py migrate

# 5. Create a superuser (optional)
python manage.py createsuperuser

# 6. Start the development server
python manage.py runserver
```

---

## 🧠 GNN Training (Optional)

If you'd like to train the GraphSAGE model from scratch:

```bash
cd recommender
python train.py  # loads data, trains model, saves embeddings
```

Later, you can use the embeddings in `recommend.py` to return product suggestions.

> Pretrained model checkpoints will be available in `/recommender/checkpoints`.

---

## 🛠 Technologies Used

- **Backend**: Django 4.x
- **ML Framework**: PyTorch, PyTorch Geometric
- **Database**: SQLite (dev) / PostgreSQL (prod-ready)
- **Frontend**: HTML, Bootstrap (optional upgrade to React/Tailwind)
- **APIs**: Django REST Framework (in progress)

---

## 📈 Example Use Cases

- AI personalization in small-to-mid eCommerce businesses
- Academic research on GNNs in recommendation systems
- Demo project for machine learning deployment in production web apps

---

## 📌 Roadmap

- [x] Develop eCommerce web service.
- [ ] GNN training module with PyTorch
- [ ] GNN inference via API
- [ ] Frontend integration with recommendation results
- [ ] Docker & deployment scripts

---


---

## 📌 Done

#### 1. Setup Backend
- [x] Created Core App.
- [x] Configured Templates.
- [x] Configure Static Files.
- [x] Configured Static and Media Files in Settings and `Urls.py`.
- [x] Create new view, urls, and template then runserver.
- [x] Configure template inheritance and partials.
- [x] Create `home_page.html`.
- [x] Create `about_page.html`.

#### 2. Setup Frontend
- [x] Created Next.js Project.
- [ ] Connect Frontend With Backend.
- [ ] Make `HomePage` and connect it with backend.


<!-- - [ ] Configured Database.
- [ ] Configured Django REST Framework.
- [ ] Configured Django PostgreSQL. -->

---

## 🤝 Contribution

We welcome collaborators! To contribute:

1. Fork the repo
2. Create a feature branch (`git checkout -b feature-name`)
3. Commit your changes
4. Push to your branch and create a PR

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](./LICENSE) file for details.

---

## 👨‍💼 Author

Developed with 💡 by **Abdalrhman Morida** from **Ai Solutions LLC** 
Connect: [📬 Email](mailto:abdalrhman30x@email.com) | [📲 Telegram](https://t.me/abdalrhman10x)
---

## ⭐ Star This Repo

If you find this project useful, give it a ⭐ to help others discover it!
```

## Dataset

📊 Your Data Summary:
Reviews Dataset
Column	Description
user_id	ID of the user who reviewed
parent_asin	ID of the product reviewed
rating	Rating score
timestamp	Time of the review
Metadata Dataset
Column	Description
parent_asin	Product ID
average_rating	Average product rating
rating_number	Number of ratings
main_category	Category of product
price, features, etc.	Other attributes you might use later

