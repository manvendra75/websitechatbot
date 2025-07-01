/**
 * HolidayMe Chat Widget
 * Easy integration chatbot for holidayme.com
 * 
 * Usage: 
 * <script src="path/to/holidayme-chat-widget.js"></script>
 * <script>HolidaymeChatWidget.init('YOUR_STREAMLIT_APP_URL');</script>
 */

window.HolidaymeChatWidget = (function() {
    'use strict';
    
    let config = {
        streamlitUrl: '',
        position: 'bottom-right',
        theme: {
            primaryColor: '#1e3a8a',
            secondaryColor: '#3b82f6',
            textColor: '#ffffff'
        },
        text: {
            buttonTooltip: 'Chat with HolidayMe Assistant',
            headerTitle: '🏖️ HolidayMe Assistant',
            headerSubtitle: 'B2B Travel Technology Solutions'
        }
    };
    
    let isInitialized = false;
    let isOpen = false;
    let elements = {};
    
    function createStyles() {
        const styles = `
            .holidayme-chat-widget * {
                box-sizing: border-box;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            }
            
            .holidayme-chat-button {
                position: fixed;
                bottom: 20px;
                right: 20px;
                width: 60px;
                height: 60px;
                background: linear-gradient(135deg, ${config.theme.primaryColor} 0%, ${config.theme.secondaryColor} 100%);
                border-radius: 50%;
                border: none;
                cursor: pointer;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
                z-index: 9998;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 24px;
                color: white;
                transition: all 0.3s ease;
                user-select: none;
            }
            
            .holidayme-chat-button:hover {
                transform: scale(1.1);
                box-shadow: 0 6px 16px rgba(0, 0, 0, 0.4);
            }
            
            .holidayme-chat-button:active {
                transform: scale(0.95);
            }
            
            .holidayme-chat-window {
                position: fixed;
                bottom: 90px;
                right: 20px;
                width: 400px;
                height: 600px;
                background: white;
                border-radius: 15px;
                box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
                z-index: 9999;
                overflow: hidden;
                transform: translateY(20px);
                opacity: 0;
                visibility: hidden;
                transition: all 0.3s ease;
                border: 2px solid #e5e7eb;
            }
            
            .holidayme-chat-window.open {
                transform: translateY(0);
                opacity: 1;
                visibility: visible;
            }
            
            .holidayme-chat-header {
                background: linear-gradient(135deg, ${config.theme.primaryColor} 0%, ${config.theme.secondaryColor} 100%);
                color: ${config.theme.textColor};
                padding: 15px 20px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .holidayme-chat-header h3 {
                margin: 0;
                font-size: 16px;
                font-weight: 600;
            }
            
            .holidayme-chat-header p {
                margin: 2px 0 0 0;
                font-size: 12px;
                opacity: 0.9;
            }
            
            .holidayme-chat-close {
                background: none;
                border: none;
                color: ${config.theme.textColor};
                font-size: 20px;
                cursor: pointer;
                padding: 5px;
                border-radius: 50%;
                width: 30px;
                height: 30px;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: background-color 0.2s ease;
            }
            
            .holidayme-chat-close:hover {
                background-color: rgba(255, 255, 255, 0.2);
            }
            
            .holidayme-chat-iframe {
                width: 100%;
                height: calc(100% - 70px);
                border: none;
                background: #f9fafb;
            }
            
            .holidayme-chat-loading {
                display: flex;
                align-items: center;
                justify-content: center;
                height: calc(100% - 70px);
                background: #f9fafb;
                font-size: 14px;
                color: #666;
            }
            
            .holidayme-chat-spinner {
                border: 2px solid #f3f3f3;
                border-top: 2px solid ${config.theme.primaryColor};
                border-radius: 50%;
                width: 20px;
                height: 20px;
                animation: holidayme-spin 1s linear infinite;
                margin-right: 10px;
            }
            
            @keyframes holidayme-spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            /* Mobile responsiveness */
            @media (max-width: 768px) {
                .holidayme-chat-window {
                    width: calc(100vw - 20px);
                    height: calc(100vh - 100px);
                    right: 10px;
                    bottom: 80px;
                    border-radius: 10px;
                }
                
                .holidayme-chat-button {
                    right: 15px;
                    bottom: 15px;
                }
            }
            
            /* Notification badge */
            .holidayme-chat-badge {
                position: absolute;
                top: -5px;
                right: -5px;
                background: #ef4444;
                color: white;
                border-radius: 50%;
                width: 20px;
                height: 20px;
                font-size: 11px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
                animation: holidayme-pulse 2s infinite;
            }
            
            @keyframes holidayme-pulse {
                0% { transform: scale(1); }
                50% { transform: scale(1.1); }
                100% { transform: scale(1); }
            }
        `;
        
        const styleElement = document.createElement('style');
        styleElement.textContent = styles;
        document.head.appendChild(styleElement);
    }
    
    function createHTML() {
        const container = document.createElement('div');
        container.className = 'holidayme-chat-widget';
        container.innerHTML = `
            <button class="holidayme-chat-button" id="holidayme-chat-button" title="${config.text.buttonTooltip}">
                🏖️
            </button>
            
            <div class="holidayme-chat-window" id="holidayme-chat-window">
                <div class="holidayme-chat-header">
                    <div>
                        <h3>${config.text.headerTitle}</h3>
                        <p>${config.text.headerSubtitle}</p>
                    </div>
                    <button class="holidayme-chat-close" id="holidayme-chat-close" title="Close chat">×</button>
                </div>
                <div class="holidayme-chat-loading" id="holidayme-chat-loading">
                    <div class="holidayme-chat-spinner"></div>
                    Loading assistant...
                </div>
                <iframe 
                    id="holidayme-chat-iframe"
                    class="holidayme-chat-iframe"
                    src="${config.streamlitUrl}" 
                    title="HolidayMe Chat Assistant"
                    allow="camera; microphone; clipboard-read; clipboard-write"
                    sandbox="allow-same-origin allow-scripts allow-forms allow-popups allow-popups-to-escape-sandbox"
                    style="display: none;">
                </iframe>
            </div>
        `;
        
        document.body.appendChild(container);
        
        // Store element references
        elements.button = document.getElementById('holidayme-chat-button');
        elements.window = document.getElementById('holidayme-chat-window');
        elements.close = document.getElementById('holidayme-chat-close');
        elements.iframe = document.getElementById('holidayme-chat-iframe');
        elements.loading = document.getElementById('holidayme-chat-loading');
    }
    
    function bindEvents() {
        // Toggle chat
        elements.button.addEventListener('click', toggleChat);
        elements.close.addEventListener('click', closeChat);
        
        // Iframe load event
        elements.iframe.addEventListener('load', function() {
            setTimeout(() => {
                elements.loading.style.display = 'none';
                elements.iframe.style.display = 'block';
            }, 2000); // Give Streamlit time to fully load
        });
        
        // Listen for messages from iframe
        window.addEventListener('message', function(event) {
            if (event.data === 'iframe-loaded') {
                elements.loading.style.display = 'none';
                elements.iframe.style.display = 'block';
            }
        });
        
        // Close on escape key
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape' && isOpen) {
                closeChat();
            }
        });
        
        // Close when clicking outside
        document.addEventListener('click', function(e) {
            if (isOpen && 
                !elements.window.contains(e.target) && 
                !elements.button.contains(e.target)) {
                closeChat();
            }
        });
    }
    
    function toggleChat() {
        if (isOpen) {
            closeChat();
        } else {
            openChat();
        }
    }
    
    function openChat() {
        elements.window.classList.add('open');
        elements.button.style.transform = 'rotate(45deg)';
        isOpen = true;
        
        // Track analytics
        trackEvent('chat_opened');
        
        // Show loading initially
        elements.loading.style.display = 'flex';
        elements.iframe.style.display = 'none';
    }
    
    function closeChat() {
        elements.window.classList.remove('open');
        elements.button.style.transform = 'rotate(0deg)';
        isOpen = false;
        
        // Track analytics
        trackEvent('chat_closed');
    }
    
    function trackEvent(eventName, data = {}) {
        // Google Analytics (if available)
        if (typeof gtag !== 'undefined') {
            gtag('event', eventName, {
                'event_category': 'HolidayMe_Chat',
                'custom_map': data
            });
        }
        
        // Console log for debugging
        console.log(`HolidayMe Chat: ${eventName}`, data);
        
        // Send to custom analytics endpoint (if needed)
        if (config.analyticsUrl) {
            fetch(config.analyticsUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    event: eventName,
                    timestamp: Date.now(),
                    url: window.location.href,
                    ...data
                })
            }).catch(e => console.warn('Analytics failed:', e));
        }
    }
    
    function showNotification() {
        if (!elements.button.querySelector('.holidayme-chat-badge')) {
            const badge = document.createElement('div');
            badge.className = 'holidayme-chat-badge';
            badge.textContent = '1';
            elements.button.appendChild(badge);
            
            // Auto-remove after 10 seconds
            setTimeout(() => {
                if (badge.parentNode) {
                    badge.parentNode.removeChild(badge);
                }
            }, 10000);
        }
    }
    
    // Public API
    return {
        init: function(streamlitUrl, options = {}) {
            if (isInitialized) {
                console.warn('HolidayMe Chat Widget already initialized');
                return;
            }
            
            if (!streamlitUrl) {
                console.error('HolidayMe Chat Widget: streamlitUrl is required');
                return;
            }
            
            // Merge options
            config = Object.assign(config, options);
            config.streamlitUrl = streamlitUrl;
            
            // Wait for DOM
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', function() {
                    createStyles();
                    createHTML();
                    bindEvents();
                    isInitialized = true;
                });
            } else {
                createStyles();
                createHTML();
                bindEvents();
                isInitialized = true;
            }
        },
        
        open: openChat,
        close: closeChat,
        toggle: toggleChat,
        showNotification: showNotification,
        
        // Configuration methods
        setTheme: function(theme) {
            config.theme = Object.assign(config.theme, theme);
        },
        
        setText: function(text) {
            config.text = Object.assign(config.text, text);
        }
    };
})();