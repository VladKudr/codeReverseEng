# Промпт: Этап 09 — Экспорт в OpenSpec specs library (для CLI-агента)

> **Пути в этом промпте относительные** (`workspace/...`). Он перенесён из
> исходного комплекта без изменений, потому что уже написан под CLI-агента.
> Перед запуском выполните агента с рабочим каталогом, в котором лежит
> `workspace/`, либо замените `workspace/` на `{WS}` тем же `sed`, что и в
> остальных этапах — см. `RUNBOOK.md`.

## Роль
Ты — системный аналитик, переносящий восстановленные требования в формат
OpenSpec (`openspec/specs/`), чтобы они стали живой specs library рядом с кодом.
Дальше AI-агенты читают эти specs как источник истины перед любым изменением.

## Ключевые принципы этапа
1. Экспортируется СИНТЕЗ (этапы 5–7: FR из SRS, stories, потоки, единая
   доменная модель), а не сырые экстракты. Сырые правила (rules.yaml) — слой
   трассировки под требованиями.
2. Два класса свидетельств, не смешивать:
   - **Код** (артефакты пайплайна) = «как работает». Только он даёт право
     на SHALL в spec.md.
   - **Документы** (`workspace/docs/`: транскрибации встреч, прошлые ТЗ) =
     «как задумывалось». Они подтверждают, объясняют «зачем», вскрывают
     противоречия и нереализованные намерения — но сами по себе НЕ порождают
     требований в specs.
3. Ничего из смыслов не терять: у каждого артефакта ниже указано его место.

## Предусловие
Выполнен `openspec init`. Документы (если есть) лежат в `workspace/docs/`
(md/txt; транскрибации и ТЗ вперемешку допустимы). Если папки нет —
проходы D пропускаются, остальное работает как обычно.

## Карта переноса смыслов
| Источник | Куда идёт |
|---|---|
| SRS раздел 4: FR по эпикам | скелет Requirements в spec.md |
| stories.md: AC (Дано/Когда/Тогда) | Scenarios (happy path) |
| rules.yaml: `edge_cases_and_errors` | Scenarios (error/boundary) |
| rules.yaml: правила high/medium | строки трассировки под Requirements |
| rules.yaml: `config_driven` ключи | пометка «задаётся конфигурацией: <ключ>» |
| rules.yaml: `actors_affected` | роли в GIVEN + словарь ролей в project.md |
| flows.md: sequence сквозных потоков | design.md, раздел «Поток» |
| Единая доменная модель, слитые сущности | design.md, раздел «Домен» |
| Статусные модели | stateDiagram в design.md + Requirement «допустимые переходы» |
| SRS раздел 6: НФТ из реализации | `specs/_cross-cutting/spec.md` с пометкой «выведено из реализации» |
| SRS разделы 1–2 | openspec/project.md |
| SRS раздел 8, противоречия, confidence:low | `specs/_open-questions.md` |
| **docs: обоснование существующего поведения** | абзац Rationale под Requirement + Purpose в spec.md; «чтобы …» в stories |
| **docs: бизнес-термины, роли, определения** | project.md (словарь) |
| **docs: утверждение, противоречащее коду** | `_open-questions.md` (обе ссылки: doc и file:line) |
| **docs: намерение, не найденное в коде** | `workspace/final/unimplemented_intent.md` — бэклог кандидатов на будущие `openspec/changes/`; в specs НЕ идёт |
| traceability.csv | + колонка `openspec_path` |

## Режим выполнения (лимит ~30–50k токенов на задачу)

### Проход D1 — экстракция из документов (фан-аут, дешёвая модель; только если есть workspace/docs/)
- **Одна задача = один документ** (длинные транскрибации режь по ~30k токенов).
- **Выход:** `workspace/docs_extracts/<doc>.yaml`:
  ```yaml
  claims:
    - id: DOC-<doc>-<NNN>
      type: behavior | rationale | term | role | constraint | intent
      statement: <одно утверждение своими словами, без интерпретации>
      quote: <короткая цитата-основание>
      source: <файл + таймкод/раздел>
      speaker_or_author: <если известно, иначе UNCERTAIN>
      date: <если известна>
  ```
- Извлекай только утверждения о системе/процессе. Мнения, спор без вывода,
  орг-вопросы — пропускай. Не решай на этом шаге, верно ли утверждение.

### Проход A — план экспорта (малый контекст)
- **Вход:** flows.md; заголовки FR из SRS; из rules.yaml — `id`, `statement`,
  `category`, `confidence`; из docs_extracts — `id`, `type`, `statement`.
- **Выход:** `workspace/final/export_plan.yaml` — capabilities (kebab-case,
  по бизнес-потокам, НЕ по репозиториям): fr_ids, story_ids, rule_ids,
  claim_ids, сущности, endpoints, фрагменты flows. Отдельно:
  `uncovered_rules`, `quarantine` (confidence: low), `cross_cutting` (НФТ),
  `unmatched_claims` (claims, не привязанные ни к одной capability).

### Проход D2 — сверка документов с кодом (по capability или пакетно; средняя модель)
Для каждого claim типа behavior/constraint против правил его capability:
- **confirms** — код и документ согласны → claim станет Rationale.
- **contradicts** — расходятся → пункт в `_open-questions.md`
  (формулировка, ссылка на doc, ссылка на file:line, вопрос владельцу).
- **not_in_code** (и type: intent) → строка в `unimplemented_intent.md`
  (намерение, источник, дата; пометь, если документ старше среза кода).
Результат: `workspace/final/claims_verdicts.yaml`. Уверенность правил
от подтверждения документом НЕ повышается — это независимая ось.

### Проход B — экспорт capability (фан-аут; синтез, модель среднего уровня и выше)
- **Вход задачи:** её фрагмент export_plan; её FR; её stories/AC; её правила
  полностью; её фрагменты домена/статусов/openapi/flows; её claims с
  вердиктом confirms (типы rationale/behavior/term).
- **Выход:** `openspec/specs/<capability>/spec.md` + `design.md`.
- Формат Requirement:
  ```markdown
  ### Requirement: <заголовок FR>
  Система SHALL <формулировка FR, наблюдаемое поведение>.

  Rationale: <«зачем» из подтверждённых claims, 1–2 предложения; если нет — опусти>

  _Trace: FR-NNN | rules: <id, id> | <file:lines> | docs: <claim_ids или —> | confidence: <min по правилам>_

  #### Scenario: <из AC>
  - GIVEN <Дано, с ролью>
  - WHEN <Когда>
  - THEN <Тогда>

  #### Scenario: <из edge_case>
  - GIVEN <контекст> / WHEN <case> / THEN <behavior из кода>
  ```
- Purpose в шапке spec.md пиши из flows + подтверждённых rationale-claims.
- `uncovered_rules` — отдельными Requirements с пометкой
  `<!-- нет FR: кандидат на ревизию SRS -->`.
- Формулировки не обогащать сверх источников; спорное — в open questions.
- design.md: «Поток» (sequence), «Домен» (erDiagram, источник истины по полям),
  «Состояния» (stateDiagram), «API» (таблица endpoints).

### Проход C — сборка и валидация (малый контекст)
- `_cross-cutting/spec.md` (НФТ), `_open-questions.md` (low-confidence,
  противоречия кода, вердикты contradicts), `unimplemented_intent.md`
  (вердикты not_in_code, отсортировать по дате, свежие сверху).
- project.md: границы, термины и роли (код + docs, при конфликте
  формулировок — код главнее, доковская в скобках), пометка
  «specs восстановлены реверсом, as-is, срез кода <repo@ref>».
- traceability.csv: колонка `openspec_path`.
- `openspec validate`; сверка покрытия.

## Самопроверка перед завершением (проход C)
- Каждый FR — ровно в одном spec.md; каждое правило high/medium — в trace
  или отдельным Requirement. Расхождения объяснены.
- Каждый edge_case стал сценарием (или явно указано, почему нет).
- Каждый claim из docs_extracts имеет вердикт (confirms/contradicts/
  not_in_code/unmatched) — потерянных claims нет.
- В spec.md нет ни одного требования, чей единственный источник — документ
  (docs без rules в trace = нарушение, перенести в unimplemented_intent).
- НФТ все в _cross-cutting; ни одного Requirement без `_Trace:`.
- `_open-questions.md` не пуст ЛИБО обосновано, почему.
- `openspec validate` без ошибок.

## Что дальше (знать, не делать)
Новые фичи — через `/openspec:propose` → delta specs → реализация → `archive`.
`unimplemented_intent.md` — готовый список тем для первых proposals.
