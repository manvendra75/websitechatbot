# Backend PDF Documents

This folder contains PDF documents that are automatically loaded by the chatbot.

## Adding PDFs

1. **Copy your PDF files to this folder:**
   ```bash
   cp your-pdf-files/*.pdf docs/
   ```

2. **Update the BACKEND_CONFIG in chatbot.py:**
   ```python
   BACKEND_CONFIG = {
       "website_url": "https://www.holidayme.com",
       "pdf_files": [
           "docs/your-file-1.pdf",
           "docs/your-file-2.pdf",
           # Add more files here
       ]
   }
   ```

3. **Commit to git:**
   ```bash
   git add docs/
   git commit -m "Add backend PDF documents"
   git push origin main
   ```

## Current Files

- Add your PDF files here
- They will be automatically loaded by the application
- Make sure to update BACKEND_CONFIG when adding new files

## Examples

Good PDF files to add:
- Service brochures (Umrah, Dubai, Maldives packages)
- Booking policies and terms
- Visa requirements and travel guides
- FAQ documents
- Price lists and package details