import asyncio
from svdb.app import AppFlow


async def main():
    """Test the vector database functionality."""
    async with AppFlow() as app:
        # Insert multiple URLs
        print("Inserting URLs...")
        urls = [
            'https://www.geeksforgeeks.org/python-typer-module/',
            'https://www.geeksforgeeks.org/top-10-system-design-interview-questions-and-answers/',
            'https://www.geeksforgeeks.org/how-to-design-a-rate-limiter-api-learn-system-design/',
            'https://www.geeksforgeeks.org/system-design-url-shortening-service/'
        ]
        
        for url in urls:
            success = await app.write_to_db_from_url(url)
            if success:
                print(f"✓ Successfully inserted: {url}")
            else:
                print(f"✗ Failed to insert: {url}")
        
        print("\n" + "="*80)
        print("Searching for closest articles...")
        print("="*80 + "\n")
        
        # Search for closest articles
        query_text = 'system design rate limiting'
        articles = await app.find_closest_articles_by_text(query_text, num_results=3)
        
        if articles:
            if not isinstance(articles, list):
                articles = [articles]
            
            print(f"Found {len(articles)} closest article(s) for query: '{query_text}'\n")
            
            for idx, article in enumerate(articles, 1):
                print(f"\n--- Result {idx} ---")
                print(f"Table: {article.get('table_name', 'N/A')}")
                print(f"ID: {article.get('id', 'N/A')}")
                print(f"Title: {article.get('title', 'N/A')}")
                print(f"URL: {article.get('url', 'N/A')}")
                print(f"Content preview: {article.get('content', 'N/A')[:200]}...")
                
                # Show embedding info
                encoded_data = article.get('encoded_data')
                if encoded_data:
                    if isinstance(encoded_data, list):
                        print(f"Embedding dimensions: {len(encoded_data)}")
                    else:
                        print(f"Embedding type: {type(encoded_data)}")
            
            # Test query by ID
            print("\n" + "="*80)
            print("Testing query by ID...")
            print("="*80 + "\n")
            
            first_article_id = articles[0].get('id')
            if first_article_id is not None:
                table_name = articles[0].get('table_name', 'articles')
                row = app.database.query_by_id(table_name, first_article_id)
                if row:
                    print(f"✓ Successfully queried by ID {first_article_id}")
                    print(f"  Title: {row.get('title', 'N/A')}")
                else:
                    print(f"✗ Failed to query by ID {first_article_id}")
        else:
            print("No articles found.")
        
        # Display timing statistics
        timing_stats = app.get_timing_stats()
        if timing_stats:
            print("\n" + "="*80)
            print("Timing Statistics")
            print("="*80)
            for key, value in timing_stats.items():
                print(f"{key}: {value:.2f} μs")


if __name__ == "__main__":
    asyncio.run(main())