# HolidayMe Chatbot Deployment Guide

## 🚀 **Phase 1: Iframe Deployment (Quick Start)**

### **Step 1: Deploy Streamlit App**

**Option A: Streamlit Cloud (Recommended for quick start)**
1. Push your code to GitHub (already done)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository: `manvendra75/websitechatbot`
4. Deploy and get your app URL: `https://your-app-name.streamlit.app`

**Option B: Heroku Deployment**
```bash
# Create Procfile
echo "web: streamlit run chatbot.py --server.port=\$PORT --server.address=0.0.0.0" > Procfile

# Deploy to Heroku
heroku create holidayme-chatbot
git push heroku main
```

**Option C: DigitalOcean App Platform**
1. Connect GitHub repository
2. Set build command: `pip install -r requirements.txt`
3. Set run command: `streamlit run chatbot.py --server.port=$PORT --server.address=0.0.0.0`

### **Step 2: Integration on HolidayMe.com**

**Method 1: Direct HTML Integration**
```html
<!-- Add to holidayme.com before closing </body> tag -->
<script>
// Load chat widget
(function() {
    var chatWidget = document.createElement('div');
    chatWidget.innerHTML = `
        <button id="holidayme-chat-btn" style="
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 60px;
            height: 60px;
            background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
            border-radius: 50%;
            border: none;
            cursor: pointer;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            z-index: 1000;
            font-size: 24px;
            color: white;
        ">🏖️</button>
        
        <div id="holidayme-chat-window" style="
            position: fixed;
            bottom: 90px;
            right: 20px;
            width: 400px;
            height: 600px;
            background: white;
            border-radius: 15px;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
            z-index: 1001;
            display: none;
            overflow: hidden;
        ">
            <div style="
                background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
                color: white;
                padding: 15px 20px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            ">
                <h3 style="margin: 0; font-size: 16px;">🏖️ HolidayMe Assistant</h3>
                <button id="holidayme-chat-close" style="
                    background: none;
                    border: none;
                    color: white;
                    font-size: 20px;
                    cursor: pointer;
                ">×</button>
            </div>
            <iframe 
                src="YOUR_STREAMLIT_APP_URL" 
                style="width: 100%; height: calc(100% - 60px); border: none;"
                title="HolidayMe Chat Assistant">
            </iframe>
        </div>
    `;
    
    document.body.appendChild(chatWidget);
    
    // Event handlers
    var isOpen = false;
    var btn = document.getElementById('holidayme-chat-btn');
    var window = document.getElementById('holidayme-chat-window');
    var close = document.getElementById('holidayme-chat-close');
    
    btn.onclick = function() {
        isOpen = !isOpen;
        window.style.display = isOpen ? 'block' : 'none';
    };
    
    close.onclick = function() {
        isOpen = false;
        window.style.display = 'none';
    };
})();
</script>
```

**Method 2: WordPress Plugin Integration**
```php
// Add to functions.php
function holidayme_chat_widget() {
    ?>
    <script src="https://your-cdn.com/holidayme-chat.js"></script>
    <script>
        new HolidaymeChatWidget({
            streamlitUrl: 'YOUR_STREAMLIT_APP_URL',
            position: 'bottom-right'
        });
    </script>
    <?php
}
add_action('wp_footer', 'holidayme_chat_widget');
```

### **Step 3: Configuration & Customization**

**Environment Variables for Production:**
```toml
# .streamlit/secrets.toml
OPENAI_API_KEY = "your-production-api-key"
ENVIRONMENT = "production"
WEBSITE_URL = "https://www.holidayme.com"
```

**Custom Branding:**
```python
# Add to chatbot.py
BRAND_CONFIG = {
    "primary_color": "#1e3a8a",  # HolidayMe blue
    "secondary_color": "#3b82f6",
    "logo_url": "https://holidayme.com/logo.png",
    "company_name": "HolidayMe"
}
```

## 🔧 **Phase 2: API Migration Planning**

### **Architecture Overview**
```
HolidayMe.com → JavaScript Widget → FastAPI Backend → Vector Database
                     ↓
              Real-time Chat Interface
```

### **Backend Structure (FastAPI)**
```python
# main.py
from fastapi import FastAPI, UploadFile, WebSocket
from pydantic import BaseModel

app = FastAPI()

class ChatMessage(BaseModel):
    message: str
    session_id: str

@app.post("/api/chat")
async def chat(message: ChatMessage):
    # Your RAG logic here
    pass

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    # Real-time chat via WebSocket
    pass
```

### **Migration Benefits**
- ✅ **Better Performance** - No Streamlit overhead
- ✅ **Real-time Chat** - WebSocket support
- ✅ **Scalability** - Handle multiple users
- ✅ **Custom UI** - Full control over chat interface
- ✅ **Analytics** - Detailed usage tracking
- ✅ **Mobile Optimization** - Responsive design

## 📊 **Monitoring & Analytics**

### **Production Metrics**
```javascript
// Track chat usage
function trackChatEvent(event, data) {
    // Google Analytics
    gtag('event', event, {
        'event_category': 'HolidayMe_Chat',
        'event_label': data.message_type,
        'value': data.session_duration
    });
    
    // Custom analytics
    fetch('/api/analytics', {
        method: 'POST',
        body: JSON.stringify({
            event: event,
            timestamp: Date.now(),
            ...data
        })
    });
}
```

### **Performance Monitoring**
- Response time tracking
- Error rate monitoring
- User engagement metrics
- PDF processing success rates

## 🛡️ **Security Considerations**

### **Production Security**
```python
# Add rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)

@app.post("/api/chat")
@limiter.limit("10/minute")
async def chat(request: Request, message: ChatMessage):
    # Rate limited chat endpoint
    pass
```

### **CORS Configuration**
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.holidayme.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

## 🚀 **Next Steps**

1. **Immediate (This Week)**
   - Deploy Streamlit app to production
   - Test iframe integration on staging site
   - Configure OpenAI API keys

2. **Short Term (2-4 Weeks)**
   - Implement analytics tracking
   - Add custom branding
   - Performance optimization

3. **Long Term (1-3 Months)**
   - Migrate to FastAPI + JavaScript widget
   - Add real-time features
   - Advanced analytics dashboard

## 📞 **Support & Maintenance**

- **Monitoring**: Set up alerts for app downtime
- **Updates**: Regular model and dependency updates
- **Backup**: Automated vector store backups
- **Scaling**: Monitor usage and scale resources

Ready to deploy! 🎯