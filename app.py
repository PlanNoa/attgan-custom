
from flask import Flask, send_file
from flask_restful import Resource, Api
from flask_restful import reqparse, request
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename
from attgan.getimage import _attgan


app = Flask(__name__)
api = Api(app)
att = _attgan()

class EditFaceAttribute(Resource):
    def post(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('style', type=str)
            parser.add_argument('image', type=FileStorage, location='files')
            args = parser.parse_args()

            _style = args['style']
            _image = args['image']
            img = ("attgan/data/img_align_celeba_png/result.png")
            _image.save(img)
            att.getimage(_style)

            return send_file("output/1.png", mimetype='image/png')
        except Exception as e:
            return {'error': str(e)}

api.add_resource(EditFaceAttribute, '/EditFaceAttribute')

if __name__ == '__main__':
    app.run(debug=True)