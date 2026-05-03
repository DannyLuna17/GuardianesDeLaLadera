# Guardianes de la Ladera

Plataforma del Data2AI Challenge para apoyar la anticipacion del riesgo de
deslizamientos en zonas de ladera en Colombia. El proyecto integra datos
abiertos oficiales, construye datasets historicos etiquetados, entrena modelos
predictivos gobernados y muestra resultados en un dashboard operativo.

## Resumen

Guardianes de la Ladera sigue una arquitectura **model-first**:

1. Ingiere y normaliza fuentes abiertas oficiales.
2. Construye variables por zona y ventana temporal.
3. Entrena y compara modelos supervisados tabulares.
4. Evalua calibracion, desempeno y estabilidad espacial/temporal.
5. Publica probabilidades, niveles de riesgo y factores explicativos en el
   dashboard.
6. Usa una capa LLM solo al final, para comunicar resultados ya calculados.

El LLM no calcula el riesgo. La decision analitica principal viene del modelo
supervisado y del pipeline de datos.

## Que incluye el repositorio

```text
.
|-- guardianes-ladera-backend/      FastAPI, SQLAlchemy, ETL, ML y gobernanza
|-- guardianes-ladera-frontend/     Dashboard React/Vite
|-- docs/                           Estado actual, roadmap y preparacion
|-- data-raw/                       Descargas locales grandes, ignoradas por Git
`-- README.md                       Este archivo
```

## Estado tecnico actual

- Backend FastAPI con rutas de dashboard, administracion, autenticacion,
  ingesta, entrenamiento, evaluacion, monitoreo y tareas de revision.
- Frontend React/Vite conectado al backend mediante una capa API tipada.
- Pipeline de inventario de deslizamientos desde UNGRD/DesInventar y SGC
  SIMMA.
- Ingesta IDEAM para catalogo de estaciones y precipitacion diaria.
- Generacion de pseudo-ausencias con politica `pseudo_absence_temporal_v1`.
- Cinco familias de modelos integradas: GLM ridge, regresion beta, GAM de
  splines aditivas, boosted-tree liviano y XGBoost.
- Flujo champion-challenger para comparar candidatos, bloquear promociones
  inseguras y abrir revision humana cuando corresponde.
- Suite backend reportada en el ultimo checkpoint: `267` pruebas pasando.

## Datos y muestra actual

La muestra modelable actual tiene `9.074` filas:

- `3.222` etiquetas positivas unicas de eventos.
- `5.852` pseudo-ausencias temporales viables.

Como inventario/prior espacial adicional, SIMMA aporta `9.177` puntos
descargados, de los cuales `9.172` fueron asociados a municipio.

El primer benchmark enriquecido con lluvia antecedente real de IDEAM cambio la
lectura del modelo: XGBoost aparece como candidato lider. Ese candidato no esta
promovido automaticamente; la promocion sigue bloqueada hasta confirmar
estabilidad multi-cohorte, calibracion y ausencia de regresiones por slices
espaciales o temporales.

## Requisitos

- Python con `uv`.
- Node.js y `npm`.
- Docker Desktop, opcional para la ruta con Compose.
- PostgreSQL/PostGIS para ejecucion containerizada o produccion; SQLite puede
  usarse como fallback local rapido en algunos flujos del backend.

## Ejecutar el backend local

```powershell
cd guardianes-ladera-backend
uv sync
uv run python scripts/import_official_structural_catalog.py
uv run uvicorn app.main:app --reload
```

Si necesitas reconstruir el bundle estructural oficial desde insumos locales:

```powershell
cd guardianes-ladera-backend
uv run python scripts/build_official_structural_bundle.py
uv run python scripts/import_official_structural_catalog.py
```

URLs locales:

- API: `http://127.0.0.1:8000`
- OpenAPI: `http://127.0.0.1:8000/docs`

El backend usa por defecto una politica `REAL_DATA_ONLY=true`. En ese modo no
debe caer en seed/demo data para simular una operacion real.

## Ejecutar el frontend local

En otra terminal, con el backend ya levantado:

```powershell
cd guardianes-ladera-frontend
npm install
npm run dev
```

Vite sirve el dashboard normalmente en `http://127.0.0.1:5173`.

Por defecto, el frontend usa el proxy `/api/*` hacia
`http://127.0.0.1:8000`. Para apuntar a otra API, configura
`VITE_API_BASE_URL`.

## Docker Compose

```powershell
cd guardianes-ladera-backend
docker compose up -d --build
```

El perfil Compose levanta PostgreSQL, ejecuta migraciones, reconstruye/importa
la base operativa y luego inicia API y worker. La API puede tardar mientras
`runtime-bootstrap` completa la ingesta oficial inicial.

## Verificacion

Backend:

```powershell
cd guardianes-ladera-backend
uv run pytest -q
```

Frontend:

```powershell
cd guardianes-ladera-frontend
npm test -- --run
npm run build
```

Smoke checks de pipelines publicos:

```powershell
cd guardianes-ladera-backend
uv run pytest tests/test_landslide_inventory.py tests/test_feature_ingestion_ideam.py -q
```

## Fuentes de datos principales

El proyecto trabaja con fuentes oficiales o abiertas:

- IDEAM: estaciones y precipitacion diaria.
- SGC SIMMA: inventario georreferenciado de movimientos en masa.
- UNGRD/DesInventar: registros historicos de emergencias.
- DANE: limites administrativos, secciones, veredas y poblacion.
- INVIAS: red vial oficial.
- IGAC/SGC: topografia, geologia y suelos.
- NASA IMERG, CHIRPS, Sentinel y Hansen: fuentes satelitales o gridded
  candidatas para enriquecer features.
