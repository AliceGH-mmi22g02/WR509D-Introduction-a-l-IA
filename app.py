from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import requests
import redis

HUGGING_FACE_API_TOKEN = ""

# Liste des sujets interdits
forbidden_topics = [
    # Religion et spiritualité
    "religion", "dieu", "jésus", "bible", "coran", "islam", "chrétien", "musulman", "bouddhisme", "athée", "spiritualité",
    "prière", "église", "mosquée", "temple", "saint", "paradis", "enfer", "âme", "péché", "salut",

    # Politique et société
    "politique", "gouvernement", "président", "élection", "vote", "parti politique", "droite", "gauche", "communisme",
    "capitalisme", "socialisme", "manifestation", "révolution", "guerre", "paix", "armée", "police", "justice", "loi",
    "avocat", "procès", "immigration", "frontière", "nationalisme", "patriotisme", "terrorisme", "conspiration",

    # Violence et comportements inappropriés
    "violence", "arme", "couteau", "pistolet", "fusil", "bombe", "attentat", "meurtre", "suicide", "agression", "harcèlement",
    "abus", "drogue", "alcool", "cigarette", "stupéfiant", "overdose", "crime", "vol", "cambriolage", "fraude",

    # Sujets sensibles ou controversés
    "racisme", "sexisme", "discrimination", "homophobie", "transphobie", "genocide", "esclavage", "colonisation", "holocauste",
    "nazi", "fascisme", "extrémisme", "propagande", "censure", "dictature", "tyrannie", "oppression", "répression",

    # Santé et sujets personnels
    "santé mentale", "dépression", "anxiété", "blessure", "maladie", "cancer", "sida", "handicap", "suicide", "thérapie",
    "médecin", "hôpital", "médicament", "vaccin", "contagion", "épidémie", "pandémie", "allergie", "handicap", "handicapé",

    # Sujets inappropriés pour un support technique
    "amour", "relation", "mariage", "divorce", "rupture", "flirt", "rencontre", "célibataire", "famille", "enfant", "parent",
    "sexualité", "sexe", "pornographie", "nudité", "intimité", "contraception", "grossesse", "avortement", "adultère",

    # Autres sujets hors contexte
    "philosophie", "métaphysique", "existence", "univers", "cosmos", "alien", "ovni", "paranormal", "fantôme", "esprit",
    "magie", "sorcellerie", "astrologie", "horoscope", "destin", "karma", "réincarnation", "médium", "voyance", "divination",
]

# Fonction pour vérifier si le message contient un sujet interdit
def is_off_topic(message: str) -> bool:
    """Vérifie si le message contient un sujet interdit."""
    message_lower = message.lower()
    return any(topic in message_lower for topic in forbidden_topics)

def load_model_and_tokenizer(model_name):
    try:
        # Chargement du modèle et du tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        print(f"Erreur lors du chargement du modèle ou du tokenizer: {e}")
        raise

def generate_response(model, tokenizer, prompt):
    try:
        # Préparation des messages pour le modèle
        messages = [
            {"role": "system", "content": """
Vous êtes un agent de support technique pour **TechBazar**, une plateforme d'e-commerce spécialisée dans les produits électroniques et high-tech. Votre rôle est d'aider les clients à résoudre leurs problèmes techniques, à répondre à leurs questions sur les produits, et à les guider dans leurs achats. Voici les instructions à suivre :

1. **Ton et attitude** :
   - Soyez **amical, professionnel et empathique**. Les clients peuvent être frustrés ou stressés, alors faites preuve de patience et de compréhension.
   - Utilisez un langage **clair et simple** pour que même les clients non techniques puissent comprendre vos explications.
   - Restez **positif** et encourageant, même si le problème est complexe.

2. **Responsabilités** :
   - Aidez les clients à résoudre leurs problèmes techniques (par exemple, configuration d'un produit, dépannage, problèmes de livraison, etc.).
   - Répondez aux questions sur les produits (caractéristiques, compatibilité, disponibilité, etc.).
   - Guidez les clients dans leur processus d'achat en leur recommandant des produits adaptés à leurs besoins.
   - Si vous ne connaissez pas la réponse à une question, dites-le honnêtement et proposez de transférer le client à un expert ou de faire des recherches supplémentaires.

3. **Règles à suivre** :
   - Ne donnez **jamais d'informations erronées**. Si vous n'êtes pas sûr, dites-le et proposez de vérifier.
   - Évitez les **jargons techniques** inutiles. Expliquez les concepts techniques de manière simple.
   - Ne discutez **pas de sujets hors de votre domaine** (par exemple, la politique, la religion, etc.).
   - Si un client est mécontent, excusez-vous poliment et proposez une solution rapide et efficace.

4. **Exemples de réponses** :
   - Pour une question sur la livraison : "La livraison standard prend généralement 3 à 5 jours ouvrables. Voulez-vous que je vérifie le statut de votre commande ?"
   - Pour un problème technique : "Avez-vous essayé de redémarrer l'appareil ? Si le problème persiste, je peux vous guider à travers les étapes de dépannage."
   - Pour une recommandation de produit : "Si vous cherchez un ordinateur portable pour le gaming, je vous recommande le modèle XYZ. Il offre une excellente performance pour les jeux récents."

5. **Informations sur TechBazar** :
   - **Produits phares** : Ordinateurs portables, smartphones, accessoires gaming, appareils photo, gadgets connectés.
   - **Services** : Livraison gratuite pour les commandes de plus de 50 €, garantie de 2 ans sur tous les produits, support technique 24/7.
   - **Politique de retour** : Les clients ont 30 jours pour retourner un produit non utilisé et dans son emballage d'origine.

6. **Objectif final** :
   - Fournir une **expérience client exceptionnelle** en résolvant les problèmes rapidement et en offrant des conseils utiles.
   - Faire en sorte que chaque client se sente **valorisé et satisfait** de son interaction avec TechBazar.
            """},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # Génération de la réponse
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    except Exception as e:
        print(f"Erreur lors de la génération de la réponse: {e}")
        raise

import requests

def get_sentiment(text, api_token):
    API_URL = "https://api-inference.huggingface.co/models/tabularisai/multilingual-sentiment-analysis"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": text
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Erreur lors de l'appel à l'API de Hugging Face: {response.status_code}, {response.text}")


def is_very_negative(sentiment_result):
    for sentiment in sentiment_result[0]:
        if sentiment["label"] == "Very Negative":
            return sentiment["score"] > 0.15
    return False


def store_in_redis(event_data):
    r = redis.Redis(host='localhost', port=6379, db=0)
    r.rpush('negative_events', event_data)

def get_sentiment(text, api_token):
    API_URL = "https://api-inference.huggingface.co/models/tabularisai/multilingual-sentiment-analysis"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": text
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Erreur lors de l'appel à l'API de Hugging Face: {response.status_code}, {response.text}")

def store_in_redis(event_data):
    r = redis.Redis(host='localhost', port=6379, db=0)
    r.rpush('negative_events', event_data)

# Configuration du middleware CORS
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Autoriser tous les domaines (à adapter en production)
    allow_credentials=True,
    allow_methods=["POST"],  # Autoriser la méthode POST
    allow_headers=["*"],
)

class PromptRequest(BaseModel):
    prompt: str

# Chargement du modèle et du tokenizer
model, tokenizer = load_model_and_tokenizer("KingNish/Qwen2.5-0.5b-Test-ft")

@app.post("/generate")
async def generate(prompt_request: PromptRequest):
    try:
        response = generate_response(model, tokenizer, prompt_request.prompt)
        
        # Obtenir le sentiment de la requête
        try:
            sentiment_result = get_sentiment(prompt_request.prompt, HUGGING_FACE_API_TOKEN)
        except Exception as e:
            print(f"Erreur lors de l'appel à l'API de sentiment : {e}")
            sentiment_result = None
        
        # Vérifier si le sentiment est très négatif
        if sentiment_result and is_very_negative(sentiment_result):
            store_in_redis(prompt_request.prompt)

        # Vérifier si la question est hors sujet
        if is_off_topic(response):
            return {"response": "Désolé, je ne peux pas répondre à cette question. Comment puis-je vous aider avec nos produits ou services ?"}
        
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)