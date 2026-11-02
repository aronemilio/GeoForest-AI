---
description: Trabajo de investigación y multi-paso con acceso amplio. Conoce que otros agentes trabajan en GeoForest-AI Huánuco al mismo tiempo; se limita a sus archivos y evita Git destructivo.
mode: subagent
permissions:
  - action: subagent
    resource: "*"
    effect: deny
  - action: shell
    resource: "*git add ."
    effect: deny
  - action: shell
    resource: "*git add -A*"
    effect: deny
  - action: shell
    resource: "*git add --all*"
    effect: deny
  - action: shell
    resource: "*git reset*"
    effect: deny
  - action: shell
    resource: "*git stash*"
    effect: deny
  - action: shell
    resource: "*git checkout*"
    effect: deny
  - action: shell
    resource: "*git restore*"
    effect: deny
  - action: shell
    resource: "*git clean*"
    effect: deny
  - action: shell
    resource: "*git switch*"
    effect: deny
  - action: shell
    resource: "*git commit*"
    effect: ask
  - action: shell
    resource: "*git push*"
    effect: ask
---
