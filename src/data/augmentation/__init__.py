"""Module tăng cường dữ liệu cho RAG chatbot.

Bao gồm 3 phương pháp (không cần LLM API):
- graph_sampler: sinh kịch bản học tập từ prerequisite graph + chương trình khung.
- cf_augment: tạo virtual student profile bằng Collaborative Filtering (SVD).
- negative_sampler: sinh các kịch bản vi phạm ràng buộc (negative samples).
"""
