# 🧠 PlayWise - Recommendation Service

Questo repository contiene il **sistema di raccomandazione** della piattaforma **PlayWise**, un'applicazione per la scoperta e la valutazione di videogiochi.

Il Recommendation Service è responsabile di generare **suggerimenti personalizzati** per ogni utente in base alle recensioni precedenti, contribuendo a migliorare l'esperienza di esplorazione del catalogo.

---

## 🔍 A cosa serve

Il servizio analizza i punteggi e i comportamenti degli utenti per consigliare videogiochi coerenti con i loro gusti. L’obiettivo è fornire raccomandazioni pertinenti che riflettano le preferenze individuali, utilizzando un approccio automatico e scalabile.

---

## 🧰 Tecnologie Utilizzate

| Categoria         | Strumento                |
|------------------|--------------------------|
| Linguaggio       | Python                   |
| Libreria ML      | [Surprise](http://surpriselib.com) (SVD) |
| API REST         | FastAPI                  |
| MLOps            | DVC, MLflow, DAGsHub     |

---

## 🧠 Tecnica di Raccomandazione

Il sistema implementa un algoritmo di **Collaborative Filtering** basato su **SVD (Singular Value Decomposition)**.  
Questa tecnica confronta gli utenti in base alle loro valutazioni e propone titoli apprezzati da utenti simili.

Il modello viene addestrato sui dati raccolti dalle recensioni della piattaforma e restituisce le migliori raccomandazioni in base allo storico personale.

---

## ⚙️ Come provarlo

Per testare il Recommendation Service **insieme all'intera piattaforma PlayWise**, segui le istruzioni di installazione presenti nel README principale:

🔗 [Vai al README di PlayWise](https://github.com/Basi-di-dati-2)

> Include istruzioni per il setup completo tramite Docker, l'avvio di tutti i microservizi e l'accesso all'interfaccia utente.

---

## 👨‍💻 Autori

| Nome | GitHub |
|------|--------|
| Francesco Alessandro Pinto | [FrancescoPinto02](https://github.com/FrancescoPinto02) |
| Francesco Maria Torino     | [FrancescoTorino1999](https://github.com/FrancescoTorino1999) |