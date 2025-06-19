import os
from dotenv import load_dotenv
load_dotenv()

# data_processing/mysql_connector.py
import mysql.connector

def get_mysql_data(
    host=None, user=None, password=None, database=None
) -> list[tuple]:
    host = host or os.getenv("MYSQL_HOST", "localhost")
    user = user or os.getenv("MYSQL_USER", "root")
    password = password or os.getenv("MYSQL_PASSWORD", "password")
    database = database or os.getenv("MYSQL_DATABASE", "insurance_bot")
    conn = mysql.connector.connect(
        host=host, user=user, password=password, database=database
    )
    cursor = conn.cursor()
    cursor.execute("SELECT id, insurer_name, policy_name, policy_type, premium, coverage_amount, sum_assured, co_payment, network_hospitals, waiting_period, maturity_benefits, return_on_maturity, entry_age_min, entry_age_max, policy_term_min, policy_term_max, renewability, claim_process, tax_benefits, eligibility, features, exclusions, add_ons_available, grace_period, free_look_period, policy_brochure_url, full_text, premium_payment_modes, policy_status, customer_rating, contact_support, covid19_coverage, policy_tags, created_at, last_updated FROM insurance_policies")
    data = cursor.fetchall()
    cursor.close()
    conn.close()
    return data

def get_mysql_connection(
    host=None, user=None, password=None, database=None
):
    host = host or os.getenv("MYSQL_HOST", "localhost")
    user = user or os.getenv("MYSQL_USER", "root")
    password = password or os.getenv("MYSQL_PASSWORD", "password")
    database = database or os.getenv("MYSQL_DATABASE", "insurance_bot")
    conn = mysql.connector.connect(
        host=host, user=user, password=password, database=database
    )
    return conn

def get_policy_brochure_url(policy_name: str):
    """Fetch the brochure URL for a given policy name from the database."""
    conn = get_mysql_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT policy_brochure_url FROM insurance_policies WHERE policy_name = %s",
        (policy_name,)
    )
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    return result[0] if result else None

def get_policy_premium(policy_name: str):
    """Fetch the premium for a given policy name from the database."""
    conn = get_mysql_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT premium FROM insurance_policies WHERE policy_name = %s",
        (policy_name,)
    )
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    return result[0] if result else None
