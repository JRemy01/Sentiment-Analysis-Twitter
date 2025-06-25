### Crée un fichier Docker
touch Dockerfile

### Construire l’image Docker
Crée un fichier Docker: 
docker build -t tweet_project -f Dockerfile .

### Lancer le projet 
docker run --rm tweet_project

### Lancer les tests unitaires
docker run --rm tweet_project pytest

