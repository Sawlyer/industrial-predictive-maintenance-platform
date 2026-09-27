# Model card

## Objectif

Estimer la durée de vie restante d'un moteur turbofan en cycles et détecter un risque de panne dans les 30 prochains cycles.

La sortie aide à prioriser une inspection. Elle ne remplace pas une décision de sécurité aéronautique ni l'avis d'un ingénieur maintenance.

## Données

- Source de référence : NASA C-MAPSS, sous-ensemble FD001.
- 100 moteurs d'entraînement et 100 moteurs de test jamais vus.
- 3 réglages de fonctionnement et 21 capteurs par cycle.
- Variables causales calculées sur fenêtres glissantes de 5, 10 et 20 cycles.
- 6 capteurs constants retirés, 15 capteurs conservés, 169 variables finales.

Le dépôt contient aussi un générateur de données synthétiques. Il sert uniquement à lancer la démonstration sans réseau et ne doit pas être utilisé pour comparer les performances publiées sur NASA C-MAPSS.

## Modèles

- Régression `HistGradientBoosting` pour la durée de vie restante.
- Classification `HistGradientBoosting` pour le risque de panne à 30 cycles.
- Référence : régression `Ridge` sur le dernier relevé uniquement.
- Validation : `GroupKFold` par moteur, pour empêcher qu'un même moteur apparaisse dans entraînement et validation.
- Cible d'entraînement plafonnée à 125 cycles ; évaluation finale faite sur les cycles restants réels.

## Performances publiées

Évaluation NASA C-MAPSS FD001 sur 100 moteurs jamais vus, à leur dernier cycle observé.

| Métrique | Référence Ridge | Modèle avec tendances |
|---|---:|---:|
| RMSE | 18,25 | **13,81** |
| Erreur absolue moyenne | 14,90 | **10,08** |
| Score NASA | 504 | **282** |
| Prédictions tardives | 53 % | **47 %** |

Le classifieur atteint un rappel de 100 % et une précision de 93 % sur l'horizon de 30 cycles dans le rapport publié.

## Limites

- FD001 représente un seul régime de fonctionnement et un seul mode de panne.
- NASA C-MAPSS est une simulation, pas une télémétrie d'exploitation réelle.
- Les coûts d'immobilisation, d'inspection et de fausse alerte ne sont pas modélisés.
- Le seuil de classification à 0,5 est un défaut ; un système réel doit le calibrer sur une matrice de coûts.
- Aucune garantie de performance sur un autre type de moteur, régime, capteur ou distribution de données.
- Les prédictions doivent être surveillées après déploiement : dérive des variables, données manquantes et changements de régime.

## Usage recommandé

Utiliser la solution comme démonstrateur de pipeline et outil de priorisation exploratoire. Revalider, calibrer et homologuer tout modèle avant usage opérationnel.
