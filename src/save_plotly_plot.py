import plotly.express as px

def savePlotlyPlot(data):
    # Create sample data for the bar graph
    # Create a Plotly Express bar graph
    print(data)
    fig = px.bar(
        data,
        x="Category",
        y="Values",
        title="Bar Graph",
        labels={"Values": "Value", "Category": "Category"},
        color="Category",
        text="Values",
    )
    fig.update_traces(textposition="outside")  # Display values outside the bars

    # Save the figure as an HTML file
    fig.write_html("bar_graph.html")
    return fig