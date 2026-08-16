document.addEventListener("DOMContentLoaded", () => {
    const chatForm = document.getElementById("chat-form");
    const chatInput = document.getElementById("chat-input");
    const chatHistory = document.getElementById("chat-history");
    const sendBtn = document.getElementById("send-btn");

    function scrollToBottom() {
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }



    function addMessage(role, content, evidence = null) {
        const msgDiv = document.createElement("div");
        msgDiv.className = `message ${role}`;
        
        let avatar = role === 'user' ? 'U' : 'M';
        
        // Use marked to parse markdown for assistant responses
        const parsedContent = (role === 'assistant' && typeof marked !== 'undefined') 
            ? marked.parse(content) 
            : `<p>${content}</p>`;

        let innerHTML = `
            <div class="avatar">${avatar}</div>
            <div class="message-content">
                ${parsedContent}
        `;

        innerHTML += `</div>`;
        msgDiv.innerHTML = innerHTML;
        chatHistory.appendChild(msgDiv);
        scrollToBottom();
    }

    function addTypingIndicator() {
        const msgDiv = document.createElement("div");
        msgDiv.className = `message assistant typing-msg`;
        msgDiv.innerHTML = `
            <div class="avatar">M</div>
            <div class="message-content">
                <div class="typing-indicator">
                    <span></span><span></span><span></span>
                </div>
            </div>
        `;
        chatHistory.appendChild(msgDiv);
        scrollToBottom();
        return msgDiv;
    }

    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const message = chatInput.value.trim();
        if (!message) return;

        // Add user message
        addMessage("user", message);
        chatInput.value = "";
        
        // Disable input while processing
        chatInput.disabled = true;
        sendBtn.disabled = true;

        const typingIndicator = addTypingIndicator();

        try {
            const response = await fetch("/api/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ message: message })
            });

            const data = await response.json();
            
            // Remove typing indicator
            typingIndicator.remove();

            if (data.error) {
                addMessage("assistant", "Error: " + data.error);
            } else {
                addMessage("assistant", data.answer, data.evidence);
            }

        } catch (error) {
            typingIndicator.remove();
            addMessage("assistant", "Network Error: Could not connect to the backend.");
        } finally {
            // Re-enable input
            chatInput.disabled = false;
            sendBtn.disabled = false;
            chatInput.focus();
        }
    });
});
