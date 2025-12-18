#!/usr/bin/env python
"""
Example script demonstrating the integration of MinerU parser with RAGAnything

This example shows how to:
1. Process documents with RAGAnything using MinerU parser
2. Perform pure text queries using aquery() method
3. Perform multimodal queries with specific multimodal content using aquery_with_multimodal() method
4. Handle different types of multimodal content (tables, equations) in queries
"""

import os
import argparse
import asyncio
import logging
import logging.config
from pathlib import Path

# Add project root directory to Python path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.ollama import ollama_embed  # change to use ollama embedding

from dotenv import load_dotenv

load_dotenv(
    dotenv_path=".env", override=True
)  # change to True to use the milvusdbs config


def configure_logging():
    """Configure logging for the application"""
    # Get log directory path from environment variable or use current directory
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, "raganything_example.log"))

    print(f"\nRAGAnything example log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    # Get log file max size and backup count from environment variables
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # Default 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default 5 backups

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )

    # Set the logger level to INFO
    logger.setLevel(logging.INFO)
    # Enable verbose debug if needed
    set_verbose_debug(os.getenv("VERBOSE", "false").lower() == "true")


async def process_with_rag(
    file_paths: list[
        str
    ],  ##################################################3333333333333333333333333
    output_dir: str,
    api_key: str,
    base_url: str = None,
    working_dir: str = None,
    parser: str = None,
):
    """
    Process document with RAGAnything

    Args:
        file_paths: list of Paths to the document
        output_dir: Output directory for RAG results
        api_key: OpenAI API key
        base_url: Optional base URL for API
        working_dir: Working directory for RAG storage
    """
    try:
        # Create RAGAnything configuration
        config = RAGAnythingConfig(
            working_dir=working_dir or "./rag_storage",
            parser=parser,  # Parser selection: mineru or docling
            parse_method="auto",  # Parse method: auto, ocr, or txt
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
            max_concurrent_files=2,  ###########################################################################
        )

        # Define LLM model function
        def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            return openai_complete_if_cache(
                "gpt-4o-mini",
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )

        # Define vision model function for image processing
        def vision_model_func(
            prompt,
            system_prompt=None,
            history_messages=[],
            image_data=None,
            messages=None,
            **kwargs,
        ):
            # If messages format is provided (for multimodal VLM enhanced query), use it directly
            if messages:
                return openai_complete_if_cache(
                    "gpt-4o",
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=messages,
                    api_key=api_key,
                    base_url=base_url,
                    **kwargs,
                )
            # Traditional single image format
            elif image_data:
                return openai_complete_if_cache(
                    "gpt-4o",
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=[
                        {"role": "system", "content": system_prompt}
                        if system_prompt
                        else None,
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_data}"
                                    },
                                },
                            ],
                        }
                        if image_data
                        else {"role": "user", "content": prompt},
                    ],
                    api_key=api_key,
                    base_url=base_url,
                    **kwargs,
                )
            # Pure text format
            else:
                return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

        # Define embedding function - using environment variables for configuration
        embedding_dim = int(os.getenv("EMBEDDING_DIM", "4096"))
        embedding_model = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:8b")

        embedding_func = EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=8192,
            func=lambda texts: ollama_embed(  # Changing from openai_embed to ollama_embed
                texts,
                embed_model=embedding_model,
                api_key=api_key,
                base_url="http://127.0.0.1:11434",
            ),
        )

        # Initialize RAGAnything with new dataclass structure
        rag = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func,
        )

        # Process document
        await rag.process_document_complete(
            file_paths=file_paths,
            output_dir=output_dir,
            parse_method="auto",  ######################################
        )

        # Query
        logger.info("\n🧠 Interactive RAGAnything Query Mode")
        logger.info(
            "Type your query below. Type 'table' or 'equation' for multimodal input. Type 'exit' to quit.\n"
        )

        while True:
            query = input("🔍 Your query: ").strip()
            if not query:
                logger.warning("⚠️ Empty query detected. Please enter a valid question.")
                continue
            if query.lower() in ["exit", "quit"]:
                logger.info("👋 Exiting interactive mode.")
                break

            if query.lower() == "table":
                logger.info("\n[Multimodal Query]: Table-based analysis")
                table_query = input("📝 Your table-related question: ").strip()
                table_data = input("📊 Paste your CSV-style table data:\n").strip()
                table_caption = input("📝 Table caption: ").strip()

                result = await rag.aquery_with_multimodal(
                    table_query,
                    multimodal_content=[
                        {
                            "type": "table",
                            "table_data": table_data,
                            "table_caption": table_caption,
                        }
                    ],
                    mode="hybrid",
                )
                logger.info(f"Answer: {result}")

            elif query.lower() == "equation":
                logger.info("\n[Multimodal Query]: Equation-based analysis")
                equation_query = input("📝 Your equation-related question: ").strip()
                latex = input("📐 Paste your LaTeX formula:\n").strip()
                equation_caption = input("📝 Equation caption: ").strip()

                result = await rag.aquery_with_multimodal(
                    equation_query,
                    multimodal_content=[
                        {
                            "type": "equation",
                            "latex": latex,
                            "equation_caption": equation_caption,
                        }
                    ],
                    mode="hybrid",
                )
                logger.info(f"Answer: {result}")

            else:
                logger.info(f"\n[Text Query]: {query}")
                result = await rag.aquery(query, mode="hybrid")
                logger.info(f"Answer: {result}")

    except Exception as e:
        logger.error(f"Error processing with RAG: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())


def main():
    """Main function to run the example"""
    parser = argparse.ArgumentParser(description="MinerU RAG Example")
    parser.add_argument("file_path", help="Path to the document to process")
    parser.add_argument(
        "--working_dir", "-w", default="./rag_storage", help="Working directory path"
    )
    parser.add_argument(
        "--output", "-o", default="./output", help="Output directory path"
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("LLM_BINDING_API_KEY"),
        help="OpenAI API key (defaults to LLM_BINDING_API_KEY env var)",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("LLM_BINDING_HOST"),
        help="Optional base URL for API",
    )
    parser.add_argument(
        "--parser",
        default=os.getenv("PARSER", "mineru"),
        help="Optional base URL for API",
    )

    args = parser.parse_args()

    # Check if API key is provided
    if not args.api_key:
        logger.error("Error: OpenAI API key is required")
        logger.error("Set api key environment variable or use --api-key option")
        return

    # Create output directory if specified
    if args.output:
        os.makedirs(args.output, exist_ok=True)

    ############################################################################################
    # Normalize file_path into a list of files
    input_path = Path(args.file_path)
    if input_path.is_dir():
        # Collect all files in the folder (non-recursive)
        file_list = [str(p) for p in input_path.iterdir() if p.is_file()]
        # If you want recursive search, replace with:
        # file_list = [str(p) for p in input_path.rglob("*") if p.is_file()]
    elif input_path.is_file():
        file_list = [str(input_path)]
    else:
        logger.error(f"Error: Path not found: {input_path}")
        return

    logger.info(f"Files to process: {file_list}")
    # Now you can pass file_list to your batch processor
    #############################################################################################3

    # Process with RAG
    asyncio.run(
        process_with_rag(
            file_list,  ########################################################################
            args.output,
            args.api_key,
            args.base_url,
            args.working_dir,
            args.parser,
        )
    )


if __name__ == "__main__":
    # Configure logging first
    configure_logging()

    print("RAGAnything Example")
    print("=" * 30)
    print("Processing document with multimodal RAG pipeline")
    print("=" * 30)

    main()
