# Déployer l'API sur Render

Ce projet est prêt à être déployé comme **Web Service Docker**. Le conteneur lance
`uvicorn` sur `0.0.0.0` et utilise le port fourni par Render. Le modèle PyTorch est
exécuté sur CPU.

## 1. Vérifier les fichiers versionnés

Avant de pousser le projet sur GitHub, vérifiez que ces deux fichiers sont bien dans
le dépôt :

- `checkpoint.pt` : poids du modèle ;
- `tokenizer.json` : tokenizer BPE.

Ils sont indispensables au démarrage de l'API. Ne les placez pas dans `.dockerignore`
et ne les excluez pas de Git. Le fichier `checkpoint.pt` actuel pèse environ 3,8 Mo,
ce qui est adapté à un dépôt Git normal.

Les fichiers ajoutés pour le déploiement sont :

- `requirements.txt` ;
- `Dockerfile` ;
- `.dockerignore` ;
- `render.yaml`.

## 2. Tester localement (facultatif mais recommandé)

Avec Docker Desktop lancé, depuis la racine du dépôt :

```bash
docker build -t traducteur-lingala .
docker run --rm -p 10000:10000 -e PORT=10000 traducteur-lingala
```

Ouvrez ensuite <http://localhost:10000/docs> ou testez :

```bash
curl http://localhost:10000/health
```

Pour arrêter le conteneur, utilisez `Ctrl+C` dans le terminal qui l'exécute.

## 3. Mettre le projet sur GitHub

Créez un dépôt GitHub puis poussez le projet, y compris les fichiers listés à
l'étape 1. À chaque nouveau `git push`, Render pourra reconstruire et déployer
l'application automatiquement.

## 4. Créer le service dans Render

1. Connectez-vous à [Render](https://dashboard.render.com/).
2. Cliquez sur **New** puis **Blueprint** et sélectionnez votre dépôt GitHub.
3. Render détecte `render.yaml`. Vérifiez le nom du service, puis cliquez sur
   **Apply**.
4. Dans les logs de déploiement, attendez le message indiquant qu'Uvicorn écoute
   sur le port. Le premier démarrage est plus long car le checkpoint est chargé en
   mémoire.
5. Lorsque le déploiement réussit, ouvrez l'URL `https://<nom-du-service>.onrender.com`.

La page web est servie à la racine. La documentation interactive est disponible sur
`/docs`, et Render vérifie automatiquement `GET /health`.

### Alternative sans Blueprint

Dans Render, choisissez **New > Web Service**, sélectionnez le dépôt et renseignez :

| Champ | Valeur |
| --- | --- |
| Language | `Docker` |
| Dockerfile Path | `./Dockerfile` |
| Health Check Path | `/health` |
| Plan | `Free` pour un essai, ou un plan avec plus de mémoire si nécessaire |

Laissez la commande Docker vide : Render utilisera la commande `CMD` du Dockerfile.

## 5. Tester l'API déployée

Remplacez `<url-render>` par l'URL fournie par Render :

```bash
curl https://<url-render>/health
curl -X POST https://<url-render>/translate \
  -H "Content-Type: application/json" \
  -d '{"sentence":"Bonjour, comment allez-vous ?","method":"beam"}'
```

## Dépannage

- **`checkpoint.pt ou tokenizer.json est introuvable`** : les fichiers ne sont pas
  présents dans le dépôt ou sont exclus du contexte Docker. Ajoutez-les à Git et
  redéployez.
- **Erreur de mémoire / service qui redémarre** : PyTorch et le modèle s'exécutent
  en RAM. Choisissez un plan Render disposant de davantage de mémoire.
- **Premier appel lent après une période d'inactivité** : les services Render Free
  peuvent s'arrêter après inactivité et doivent redémarrer ; c'est normal.
- **Échec de la vérification de santé** : consultez les logs. L'endpoint `/health`
  ne répond qu'après le chargement réussi du modèle.

## Mises à jour

Modifiez le code, validez puis poussez vos changements sur la branche connectée.
Avec `autoDeployTrigger: commit`, Render lance un nouveau déploiement à chaque
nouveau commit. Les logs sont accessibles dans la page **Deploys** du service.
