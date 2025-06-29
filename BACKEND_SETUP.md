# Backend Content Management

## 🎯 **Clean Frontend - No Upload Options**

The frontend now focuses purely on chat functionality. All content sources are managed in the backend configuration.

## 📁 **Adding PDF Documents to Backend**

### **Step 1: Upload PDFs to Server**

**For Streamlit Cloud:**
```bash
# Create a docs folder in your repository
mkdir docs
# Add your PDF files to the docs folder
cp your-pdfs/*.pdf docs/
# Commit to GitHub
git add docs/
git commit -m "Add backend PDF documents"
git push origin main
```

**For Server Deployment:**
```bash
# Create directory on server
mkdir /app/data/pdfs/
# Upload your PDFs
scp your-pdfs/*.pdf user@server:/app/data/pdfs/
```

### **Step 2: Update Backend Configuration**

Edit `chatbot.py` and update the `BACKEND_CONFIG`:

```python
BACKEND_CONFIG = {
    "website_url": "https://www.holidayme.com",
    "pdf_files": [
        "docs/umrah-services.pdf",           # Umrah travel packages
        "docs/dubai-packages.pdf",          # Dubai travel info
        "docs/maldives-resorts.pdf",        # Maldives resort details
        "docs/visa-requirements.pdf",       # Visa and travel docs
        "docs/booking-policies.pdf",        # Terms and conditions
    ]
}
```

### **Step 3: File Structure**
```
your-project/
├── chatbot.py
├── requirements.txt
├── .streamlit/
│   └── secrets.toml
└── docs/                    # Backend PDF storage
    ├── umrah-services.pdf
    ├── dubai-packages.pdf
    ├── maldives-resorts.pdf
    ├── visa-requirements.pdf
    └── booking-policies.pdf
```

## 🔧 **Configuration Options**

### **Environment-based Configuration**
```python
import os

# Different configs for different environments
if os.getenv("ENVIRONMENT") == "production":
    BACKEND_CONFIG = {
        "website_url": "https://www.holidayme.com",
        "pdf_files": [
            "/app/data/pdfs/umrah-services.pdf",
            "/app/data/pdfs/dubai-packages.pdf",
        ]
    }
else:
    # Development configuration
    BACKEND_CONFIG = {
        "website_url": "https://www.holidayme.com",
        "pdf_files": [
            "docs/test-document.pdf",
        ]
    }
```

### **Admin Panel Control**
Add to your `.streamlit/secrets.toml`:
```toml
SHOW_ADMIN_PANEL = true  # Set to false for production
OPENAI_API_KEY = "your-api-key"
```

## 📊 **Content Management Best Practices**

### **PDF Organization**
```
docs/
├── services/
│   ├── umrah-packages.pdf
│   ├── dubai-tours.pdf
│   └── maldives-resorts.pdf
├── policies/
│   ├── booking-terms.pdf
│   ├── cancellation-policy.pdf
│   └── privacy-policy.pdf
└── guides/
    ├── visa-requirements.pdf
    ├── travel-tips.pdf
    └── packing-checklist.pdf
```

### **Update Configuration**
```python
BACKEND_CONFIG = {
    "website_url": "https://www.holidayme.com",
    "pdf_files": [
        # Service documents
        "docs/services/umrah-packages.pdf",
        "docs/services/dubai-tours.pdf",
        "docs/services/maldives-resorts.pdf",
        
        # Policy documents
        "docs/policies/booking-terms.pdf",
        "docs/policies/cancellation-policy.pdf",
        
        # Travel guides
        "docs/guides/visa-requirements.pdf",
        "docs/guides/travel-tips.pdf",
    ]
}
```

## 🔄 **Content Updates**

### **Adding New PDFs**
1. Upload PDF to the docs folder
2. Add path to `BACKEND_CONFIG["pdf_files"]`
3. Restart the application (content will reload automatically)

### **Removing PDFs**
1. Remove path from `BACKEND_CONFIG["pdf_files"]`
2. Restart the application
3. Optionally delete the PDF file

### **Updating Existing PDFs**
1. Replace the PDF file with the new version (same filename)
2. Clear the cache by restarting or using admin panel
3. New content will be loaded automatically

## 🚀 **Deployment Workflow**

### **Production Deployment**
```bash
# 1. Add PDFs to repository
git add docs/
git commit -m "Update PDF content"

# 2. Update configuration
git add chatbot.py
git commit -m "Update backend PDF configuration"

# 3. Deploy
git push origin main
# App will automatically restart and load new content
```

### **Content Validation**
```python
# Add validation function
def validate_backend_config():
    """Validate that all configured PDFs exist"""
    missing_files = []
    for pdf_path in BACKEND_CONFIG["pdf_files"]:
        if not os.path.exists(pdf_path):
            missing_files.append(pdf_path)
    
    if missing_files:
        st.error(f"Missing PDF files: {missing_files}")
        return False
    return True
```

## 💡 **Benefits of Backend Management**

✅ **Clean User Interface** - No upload clutter  
✅ **Consistent Content** - All users see the same information  
✅ **Better Performance** - Pre-loaded and cached content  
✅ **Version Control** - PDF content tracked in Git  
✅ **Easy Updates** - Update once, applies to all users  
✅ **Professional Look** - Production-ready appearance  

## 🛠️ **Troubleshooting**

**PDFs not loading?**
- Check file paths in `BACKEND_CONFIG`
- Verify PDF files exist in the specified locations
- Check file permissions (readable by application)

**Content not updating?**
- Use admin panel "Reload Content" button
- Restart the application
- Clear browser cache

**Error loading PDFs?**
- Check PDF file format (some encrypted PDFs may fail)
- Verify PyPDF2 can read the files
- Check file size (very large PDFs may timeout)

Ready for clean, backend-managed content! 🎯