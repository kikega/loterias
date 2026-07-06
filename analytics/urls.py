from django.urls import path

from . import views

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("dashboard-content/", views.dashboard_content, name="dashboard_content"),
    path("api/datos/", views.dashboard_json, name="dashboard_json"),
    path("insertar/", views.insertar_sorteo, name="insertar_sorteo"),
    path("predicciones/", views.panel_predicciones, name="panel_predicciones"),
    path("predicciones/content/", views.predicciones_content, name="predicciones_content"),
    path("predicciones/generar/", views.generar_predicciones_view, name="generar_predicciones"),
    path("predicciones/reentrenar/", views.reentrenar_modelo_view, name="reentrenar_modelo"),
    path("login/", views.login_view, name="login"),
    path("logout/", views.logout_view, name="logout"),
]
