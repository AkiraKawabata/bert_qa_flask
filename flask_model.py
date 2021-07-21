from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String
Base = declarative_base()
class QA_data(Base):
    __tablename__='qa_data'

    id = Column(Integer, primary_key = True)
    query = Column(String)
    context = Column(String)
    answer = Column(String)