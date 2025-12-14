# Chatbot Fixes and Troubleshooting Guide

## Changes Made

### 1. **Backend (`backend/main.py`)**
   - ✅ Fixed Gemini API call to include ALL messages (not just history)
   - ✅ Added comprehensive system prompt with all your specified rules
   - ✅ Improved error handling with detailed logging
   - ✅ Fixed ExplanationData to use proper Pydantic models
   - ✅ Added health check endpoint at `/health`
   - ✅ Better timeout and error messages

### 2. **Frontend (`src/components/SeaSageChatbot.tsx`)**
   - ✅ Improved error messages to show what's actually wrong
   - ✅ Better connection error detection
   - ✅ Updated welcome message
   - ✅ Added system prompt to fallback Gemini call

### 3. **Supabase Function (`supabase/functions/chat/index.ts`)**
   - ✅ Updated system prompt to match your requirements

## How to Test

### Step 1: Start the Backend
```bash
cd backend
uvicorn main:app --reload
```

Or from project root:
```bash
uvicorn backend.main:app --reload
```

### Step 2: Check Health Endpoint
Open browser or use curl:
```bash
curl http://localhost:8000/health
```

Should return:
```json
{
  "status": "healthy",
  "rf_model_loaded": true,
  "explainer_loaded": true/false,
  "gemini_key_configured": true/false
}
```

### Step 3: Test Chat Endpoint
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is eDNA?"}
    ],
    "sessionId": "test-123"
  }'
```

### Step 4: Check Environment Variables
Make sure you have a `.env` file in the backend directory with:
```
VITE_GEMINI_API_KEY=your_api_key_here
```
OR
```
GEMINI_API_KEY=your_api_key_here
```

## Common Issues and Solutions

### Issue 1: "Cannot connect to backend"
**Solution**: Make sure the backend is running on port 8000
```bash
# Check if port 8000 is in use
netstat -ano | findstr :8000  # Windows
lsof -i :8000  # Mac/Linux
```

### Issue 2: "Gemini API Key not configured"
**Solution**: 
1. Create a `.env` file in the `backend/` directory
2. Add: `VITE_GEMINI_API_KEY=your_key_here`
3. Restart the backend server

### Issue 3: "Backend error 500"
**Solution**: Check the backend logs for detailed error messages. Common causes:
- Missing dependencies (install with `pip install -r requirements.txt`)
- SHAP not installed (optional, but recommended)
- Invalid API key format

### Issue 4: Chatbot not following rules
**Solution**: 
- The system prompt is now comprehensive and should enforce all rules
- If still not working, check backend logs to see if Gemini API is being called
- Verify the API key is valid and has quota

## Testing the Chatbot Behavior

### Test 1: General Knowledge Mode
Ask: "What is eDNA?"
Expected: Should answer using general knowledge, NOT mention any analysis results

### Test 2: Result-Aware Mode
First provide context with results, then ask: "What species were detected?"
Expected: Should only use provided data, not invent species

### Test 3: No Results Available
Ask: "What species were detected?" (without providing results)
Expected: Should say "Please upload and analyze a file first so I can answer that."

### Test 4: Honesty Check
Ask about something not in context
Expected: Should say "I don't have enough data" or similar, NOT invent facts

## Debugging

### Check Backend Logs
The backend now has detailed logging. Look for:
- `INFO: Routing to GENERAL path: ...` - Shows which path was taken
- `INFO: Calling Gemini API with X message parts` - Confirms API call
- `ERROR: Gemini API Error: ...` - Shows what went wrong

### Check Browser Console
Open browser DevTools (F12) and check Console tab for:
- Network errors
- Backend connection issues
- API response errors

### Test Backend Directly
Use the health endpoint to verify backend is running:
```bash
curl http://localhost:8000/health
```

## Next Steps

1. **Start the backend**: `uvicorn backend.main:app --reload`
2. **Start the frontend**: `npm run dev` or `bun run dev`
3. **Test with a simple question**: "What is eDNA?"
4. **Check the browser console** for any errors
5. **Check backend terminal** for detailed logs

If still not working, share:
- Backend terminal output
- Browser console errors
- Response from `/health` endpoint
- What happens when you send a message

