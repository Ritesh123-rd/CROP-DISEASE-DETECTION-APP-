from database import create_user, get_user_by_email, update_user, delete_user, init_db

def run_tutorial():
    print("--- 1. Initialize DB ---")
    init_db()
    
    print("\n--- 2. Create User (C) ---")
    email = "test_crud@example.com"
    # Delete if exists from previous run
    existing = get_user_by_email(email)
    if existing:
        delete_user(existing['id'])
        print("Cleaned up previous test user.")

    result = create_user("Test User", email, "password123", "9876543210")
    print(f"Create Result: {result}")
    
    if not result['success']:
        return

    user_id = result['user_id']
    
    print("\n--- 3. Read User (R) ---")
    user = get_user_by_email(email)
    print(f"Read User: {user}")
    
    print("\n--- 4. Update User (U) ---")
    update_res = update_user(user_id, name="Updated Name", phone="1112223333")
    print(f"Update Result: {update_res}")
    
    updated_user = get_user_by_email(email)
    print(f"User after update: {updated_user}")
    
    print("\n--- 5. Delete User (D) ---")
    del_res = delete_user(user_id)
    print(f"Delete Result: {del_res}")
    
    final_check = get_user_by_email(email)
    print(f"User after delete: {final_check}")

if __name__ == "__main__":
    run_tutorial()
