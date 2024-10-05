from sqlalchemy import Table, Column, String, DateTime, LargeBinary
from sqlalchemy.sql.sqltypes import Integer
from config.db import meta
from datetime import datetime, timezone

lecture_materials = Table(
    'lecture_materials', meta,
    Column('id', Integer, primary_key=True),
    # Column('file', String(255)), 
    Column('file_name', String(255)),  
    Column('file_type', String(50)), 
    Column('uploaded_at', DateTime, default=lambda: datetime.now(timezone.utc))
)
