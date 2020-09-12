import plotly_express as px
gapminder = px.data.gapminder()
gapminder2007 = gapminder.query("year == 2007")

px.scatter(gapminder2007, x="gdpPercap", y="lifeExp")