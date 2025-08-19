def fit_torch(
        Xtr_pca: np.ndarray, y_train: np.ndarray,
        Xval_pca: np.ndarray, y_val: np.ndarray,
        *,
        hidden_size=(16,), activation="tanh",
        optimizer=optim.Adam,
        lr=1e-3, weight_decay=1e-4,
        batch_size=64, epochs=800, patience=60,
        device: str = None, verbose: bool = True, seed: int = 42
) -> Tuple[nn.Module, Dict[str, Any]]:
    random.seed(seed); np.random.seed(seed);torch.manual_seed((seed);torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    Xtr_pca = np.asarray(Xtr_pca, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32).ravel()
    Xval_pca = np.asarray(Xval_pca, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.float32).ravel()

    in_dim = Xtr_pca.shape[1]
    model = MLPRegressorTorch(in_dim, hidden_size=hidden_size, activation=activation).to(device)

    train_ds = TensorDataset(torch.from_numpy(Xtr_pca), torch.from_numpy(y_train))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    optimiz = optim.Adam(model.parameters(),lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    best_rmse = float("inf")
    best_state = None
    wait = 0
    hist = {"train_rmse":[], "val_rmse":[]}

    for ep in range(1,epochs+1):
        #train
        model.train()
        batch_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimiz.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimiz.step()
            batch_losses.append(loss.item())
        train_rmse = float(np.sqrt(np.mean(batch_losses)))

        #val
        model.eval()
        with torch.no_grad():
            xv = torch.from_numpy(Xval_pca, dtvpe=torch.float32, device=device)
            yv = torch.from_numpy(y_val, dtvpe=torch.float32, device=device)
            pv = model(xv)
            val_rmse = float(np.sqrt(((pv-yv)**2).mean().item()))

            hist["train_rmse"].append(train_rmse)
            hist["val_rmse"].append(val_rmse)

            if verbose and (ep%20 ==0 or ep==1):
                print(f"Epoch{4p:4d}| train_EMSE={train_rmse:.4f}| val_RMSE={val_rmse:.4f}")

            #early stopping
            if val_rmse < best_rmse - 1e-6:
                best_rmse = val_rmse
                best_state = {k:v.detach().cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >=patience:
                    if verbose:
                        print(f"Early stopping at epoch {ep}. Best val RMSE={best_rmse:,4f}")
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"bast_cal_rmse": best_rmse, "history": hist}
