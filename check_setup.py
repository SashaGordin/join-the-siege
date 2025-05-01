#!/usr/bin/env python3
import os
import sys
from dotenv import load_dotenv
import redis
from openai import OpenAI
import magic
import pytesseract

def check_env_variables():
    """Check if all required environment variables are set."""
    load_dotenv()
    required_vars = ['OPENAI_API_KEY', 'REDIS_URL']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        return False
    print("✅ Environment variables check passed")
    return True

def check_redis_connection():
    """Check if Redis connection works."""
    try:
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        r = redis.from_url(redis_url)
        r.ping()
        print("✅ Redis connection successful")
        return True
    except redis.ConnectionError as e:
        print(f"❌ Redis connection failed: {str(e)}")
        return False

def check_openai_api():
    """Check if OpenAI API key is valid."""
    try:
        client = OpenAI()
        # Make a simple API call
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello!"}],
            max_tokens=5
        )
        print("✅ OpenAI API connection successful")
        return True
    except Exception as e:
        print(f"❌ OpenAI API check failed: {str(e)}")
        return False

def check_system_dependencies():
    """Check if required system dependencies are installed."""
    success = True

    # Check tesseract
    try:
        pytesseract.get_tesseract_version()
        print("✅ Tesseract OCR installed")
    except Exception as e:
        print(f"❌ Tesseract OCR not found: {str(e)}")
        success = False

    # Check libmagic
    try:
        magic.Magic()
        print("✅ libmagic installed")
    except Exception as e:
        print(f"❌ libmagic not found: {str(e)}")
        success = False

    return success

def main():
    """Run all checks."""
    print("\n🔍 Checking setup...\n")

    checks = [
        check_env_variables(),
        check_redis_connection(),
        check_openai_api(),
        check_system_dependencies()
    ]

    print("\n📝 Summary:")
    if all(checks):
        print("✅ All checks passed! You're ready to run the application.")
        return 0
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())