import kaggle
kaggle.api.authenticate()
try:
    response = kaggle.api.kernels_push('d:/Work/trading_bot/notebooks')
    print("Response:", response)
except Exception as e:
    import traceback
    traceback.print_exc()
