from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from databases import session, User, UserReview
from auth import get_password_hash, verify_password, create_access_token, get_current_user

router = APIRouter()

class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str

class LoginRequest(BaseModel):
    name: str
    password: str

@router.post("/register")
def register_user(payload: RegisterRequest):
    existing_user = session.query(User).filter(User.name == payload.name).first()
    if existing_user:   
        raise HTTPException(status_code=400, detail="Name already registered")

    last_user_id = session.query(UserReview.user_id).order_by(UserReview.user_id.desc()).first()
    new_user_id = (last_user_id[0] + 1) if last_user_id else 1  # Perhatikan ini pakai [0] karena hasilnya tuple


    hashed_password = get_password_hash(payload.password)
    new_user = User(
        name=payload.name,
        password=hashed_password,
        email=payload.email,
        user_id=new_user_id
    )
    session.add(new_user)
    session.commit()
    token = create_access_token(data={"sub": new_user.name})
    return {"token": token, "user_id": new_user.user_id, "name": new_user.name}

@router.post("/login")
def login_user(payload: LoginRequest):
    user = session.query(User).filter(User.name == payload.name).first()
    if not user or not verify_password(payload.password, user.password):
        raise HTTPException(status_code=401, detail="Invalid name or password")

    token = create_access_token(data={"sub": user.name})
    return {"token": token, "user_id": user.user_id, "name": user.name}

@router.get("/me")
def read_current_user(current_user: User = Depends(get_current_user)):
    return {
        "user_id": current_user.user_id,
        "name": current_user.name,
    }



