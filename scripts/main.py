import argparse
from src.retrieval_qa.main import upload_pdfs_to_vector_store, create_query_interface


def main():
    parser = argparse.ArgumentParser(description="FDA Regulations QA CLI")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Upload command
    upload_parser = subparsers.add_parser("upload", help="Upload PDFs to vector store")
    upload_parser.add_argument("--pdf_dir", required=True, help="Path to raw PDFs")
    upload_parser.add_argument("--store_name", required=True, help="Vector store name")

    # Single query (search only)
    search_parser = subparsers.add_parser("search", help="Run file search without LLM")
    search_parser.add_argument("--query", required=True)
    search_parser.add_argument("--store_id", required=True)

    # Single query (LLM + RAG)
    ask_parser = subparsers.add_parser("ask", help="Ask question using LLM + file search")
    ask_parser.add_argument("--query", required=True)
    ask_parser.add_argument("--store_id", required=True)

    # Batch queries
    batch_parser = subparsers.add_parser("batch", help="Run batch LLM queries")
    batch_parser.add_argument("--queries", nargs="+", required=True, help="List of questions")
    batch_parser.add_argument("--store_id", required=True)

    args = parser.parse_args()

    if args.command == "upload":
        result = upload_pdfs_to_vector_store(args.pdf_dir, args.store_name)
        print(f"Uploaded to vector store: {result['vector_store']['id']}")

    elif args.command == "search":
        interface = create_query_interface(args.store_id)
        results = interface.search_only(args.query)
        for result in results["results"]:
            print(f"{result['content_length']} characters from {result['filename']} "
                  f"(score: {result['score']:.3f})")

    elif args.command == "ask":
        interface = create_query_interface(args.store_id)
        response = interface.ask_with_llm(args.query)
        print(f"Response: {response['response']}")
        print(f"Files used: {response['files_used']}")

    elif args.command == "batch":
        interface = create_query_interface(args.store_id)
        results = interface.batch_query(args.queries, use_llm=True)
        for i, result in enumerate(results):
            print(f"Q: {args.queries[i]}")
            print(f"A: {result['response']}")
            print(f"Files used: {result['files_used']}")
            print()


if __name__ == "__main__":
    main()
