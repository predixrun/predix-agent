from fastapi.responses import JSONResponse


class TemplateJSONResponse(JSONResponse):
    def render(self, content: any) -> bytes:
        if isinstance(content, dict) and "status" in content and "data" in content:
            response_content = content
        else:
            response_content = {
                "status": "SUCCESS",
                "errorCode": "",
                "data": content
            }
        return super().render(response_content)
