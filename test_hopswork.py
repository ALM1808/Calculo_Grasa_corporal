# test_hopswork.py

import os
from dotenv import load_dotenv
import hopsworks

# ✅ Carga las variables desde el archivo .env
load_dotenv()

# 🔐 Lee las credenciales y configuración
api_key = os.getenv("HOPSWORKS_API_KEY")
project_name = os.getenv("HOPSWORKS_PROJECT")
host = os.getenv("HOPSWORKS_HOST", "https://c.app.hopsworks.ai")

# ✅ Verifica que las variables estén presentes
if not api_key or not project_name:
    raise ValueError("❌ API key o nombre del proyecto no están definidos en el archivo .env")

# 🚀 Intenta conectarse a Hopsworks
print("🔐 Conectando a Hopsworks...")

project = hopsworks.login(
    api_key_value=api_key,  # <-- nombre correcto del parámetro
    project=project_name,
    host=host
)

print(f"✅ Conexión exitosa al proyecto: {project.name}")
