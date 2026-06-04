import os
from flask import jsonify, send_from_directory

UPLOAD_FOLDER = 'uploads'  # adapte si ton chemin est différent

# ── GET /api/signs/:label/images ─────────────────────────────────────────────
@app.route('/api/signs/<label>/images', methods=['GET'])
def get_sign_images(label):
    label_upper = label.strip().upper()
    folder_path = os.path.join(UPLOAD_FOLDER, label_upper)

    if not os.path.isdir(folder_path):
        return jsonify({ 'images': [] }), 200

    allowed_ext = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
    files = [
        f for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in allowed_ext
    ]
    files.sort()

    # Retourne les URLs publiques des images
    base_url = request.host_url.rstrip('/')  
    images = [f'{base_url}/api/signs/{label_upper}/images/{f}' for f in files]

    return jsonify({ 'images': images }), 200


# ── GET /api/signs/:label/images/:filename ────────────────────────────────────
# Sert le fichier image directement
@app.route('/api/signs/<label>/images/<filename>', methods=['GET'])
def serve_sign_image(label, filename):
    label_upper = label.strip().upper()
    folder_path = os.path.join(UPLOAD_FOLDER, label_upper)
    return send_from_directory(folder_path, filename)