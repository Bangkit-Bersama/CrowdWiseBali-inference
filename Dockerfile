FROM tensorflow/tensorflow:2.18.0

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv --system-site-packages $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip3 install flask flask_cors pandas scikit-learn

WORKDIR /app

COPY . .

EXPOSE 3000

CMD [ "python3", "app.py" ]
