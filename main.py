
from dotenv import load_dotenv

load_dotenv()

from graph.graph import app

if __name__ == "__main__":
    print("Hello Advanced RAG")

    result = app.invoke(
        input={"question": "what is agent memory?"}
    )

    print("\n--- FINAL ANSWER ---")
    print(result["generation"])
