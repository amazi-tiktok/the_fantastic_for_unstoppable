# 📖 API Documentation: Fantastic For Review Moderation

## Base URL

```
http://localhost:5000/
```

---

## 1. `GET /`

**Description:**  
Returns the main demo HTML page (UI).

**Request:**  
No parameters.

**Response:**  
HTML page.

---

## 2. `POST /analyze`

**Description:**  
Analyzes a review for policy violations.

**Request Body (JSON):**
```json
{
  "review_data": {
    "text": "The review text here",
    "rating": 4,
    "name": "Reviewer Name",
    "image_url": "https://example.com/image.jpg"
  },
  "store_info": {
    "name": "Store Name",
    "category": ["Restaurant"],
    "description": "A nice place for food."
  },
  "gcom_id": "optional-id"
}
```

**Response (JSON):**
```json
{
  "overall_violation_score": 0.23,
  "action": "APPROVE",
  "action_reason": "Low violation score - appears legitimate",
  "policy_violations": {
    "advertisement": { "score": 0.0, "violations": [], "weight": 0.2 },
    "relevancy": { "score": 0.0, "violations": [], "weight": 0.15 },
    "visit_authenticity": { "score": 0.0, "violations": [], "weight": 0.1 },
    "quality": { "score": 0.0, "violations": [], "weight": 0.1 },
    "offensive": { "score": 0.0, "violations": [], "weight": 0.3 },
    "image_analysis": { "score": 0.0, "violations": [], "weight": 0.15 }
  },
  "metadata": {
    "reviewer_name": "Reviewer Name",
    "review_length": 5,
    "rating": 4,
    "has_image": true,
    "image_url": "https://example.com/image.jpg",
    "store_name": "Store Name",
    "store_category": "Restaurant",
    "analysis_timestamp": "2025-08-31T12:34:56.789Z",
    "weights_used": { "...": "..." }
  },
  "gcom_id": "optional-id"
}
```

**Possible `action` values:**  
- `"APPROVE"`: Low violation score  
- `"FLAG"`: Medium violation score (requires manual review)  
- `"REMOVE"`: High violation score (likely policy violation)

**Error Responses:**
- `400`: Missing required fields (`review_data.text` or `store_info.name`)
- `500`: Internal error

---

## 3. `POST /feedback`

**Description:**  
Submit user feedback on the moderation result.

**Request Body (JSON):**
```json
{
  "feedback": "accurate" | "inaccurate",
  "results": { ... }  // The full results object returned by /analyze
}
```

**Response (JSON):**
```json
{ "status": "success" }
```

**Error Responses:**
- `400`: Missing feedback or results

---

## 4. `GET /health`

**Description:**  
Health check endpoint.

**Request:**  
No parameters.

**Response (JSON):**
```json
{
  "status": "healthy",
  "timestamp": "2025-08-31T12:34:56.789Z",
  "version": "1.2.0"
}
```

---

## Notes

- All endpoints return JSON except `/` (HTML).
- All times are in ISO 8601 format.
- Feedback is logged to `feedback_log.csv` on the server.

---