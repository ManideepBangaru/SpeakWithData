import plotly.graph_objects as go

def prepareOutput(response_ : str, df):
    if isinstance(response_, str):
        # Check if the response looks like code for a Plotly chart
        if "px." in response_ or response_.startswith("import plotly") or "go.Figure" in response_:
            try:
                # Execute the code string to create a Plotly figure
                local_vars = {df}
                exec(response_, {"go": go}, local_vars)
                figure = local_vars.get("fig")  # Assuming the Plotly code assigns to 'fig'
                if figure:
                    return None, figure
                else:
                    return "Error: Plotly figure was not generated.", None
            except Exception as e:
                return f"Error rendering Plotly chart: {e}", None
        else:
            # Treat the response as plain text
            return response_, None
    else:
        return "Unexpected response format.", None