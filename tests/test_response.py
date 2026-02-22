from app.utils.response import success_response, error_response


def test_success_response_with_data():
    result = success_response(data={"key": "value"})
    assert result == {"status": "success", "data": {"key": "value"}, "message": None}


def test_success_response_with_message():
    result = success_response(data=None, message="Operazione completata")
    assert result == {"status": "success", "data": None, "message": "Operazione completata"}


def test_error_response():
    result = error_response("Errore di validazione")
    assert result == {"status": "error", "data": None, "message": "Errore di validazione"}


def test_error_response_with_data():
    result = error_response("Errore", data={"field": "username"})
    assert result == {"status": "error", "data": {"field": "username"}, "message": "Errore"}
