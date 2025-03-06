# puzzle_solver
## Détails des semaines :
Semaine 1 : Prise de photos, documentation, organisation du projet (tâches,...) - Aquisition
Semaine 2 : Détections des pièces et découpage (formes) + Détections des contours des pièces (trou, têtes, points) - Calibration et detouring
Semaine 3 : Détections de la couleur des pièces (couleurs) - Detouring
Semaine 4 : Matchings + Descripteur (pièces les plus compatible)
Semaine 5 : Reconstruction, corrections 
Semaine 6 : Construction de l'image
semaine 7 : - 

# Contexte : 
- Résoudre un puzzle pris depuis une photo (photo du dessus (sur fond unique), éparpillées)

# Analyse : 
- Prise de photos (uni, lumières, calibration)
- Discerner pièce (image des pièces, découpage, etc) - Detouring
- Classement des pièces et les comparer (forme, couleurs et rotation) = Descripteurs
  - Forme : Matcher les bords (les bords (keypoints), notations de chaque bords, essayer toutes les possibilités)
  - Couleur : Couleurs des bords et matching
- Reconstruction prendre le meilleur correcte (le best doit être garantie) -> ajout de contraintes (pièce de bords) + utilisation de Backtracking (si rien ne marche, triage par matching)
- Construction de l'image (rotation,...)
