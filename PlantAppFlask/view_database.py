"""
Database Viewer - Check all data in PlantCare AI database
"""

import sqlite3

DB_PATH = 'plantcare.db'

def view_all():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print("=" * 60)
    print("📊 PLANTCARE AI DATABASE VIEWER")
    print("=" * 60)
    
    # Users
    print("\n👤 USERS TABLE:")
    print("-" * 40)
    cursor.execute("SELECT * FROM users")
    users = cursor.fetchall()
    if users:
        for user in users:
            print(f"  ID: {user[0]}, Name: {user[1]}, Email: {user[2]}")
    else:
        print("  (No users yet)")
    
    # Diagnosis History
    print("\n🌿 DIAGNOSIS HISTORY:")
    print("-" * 40)
    cursor.execute("SELECT * FROM diagnosis_history ORDER BY created_at DESC LIMIT 10")
    history = cursor.fetchall()
    if history:
        for h in history:
            print(f"  ID: {h[0]}, Plant: {h[2]}, Disease: {h[3]}, Confidence: {h[4]}")
            print(f"     Date: {h[9]}")
    else:
        print("  (No diagnoses yet)")
    
    # Chat History
    print("\n💬 CHAT HISTORY:")
    print("-" * 40)
    cursor.execute("SELECT * FROM chat_history ORDER BY created_at DESC LIMIT 5")
    chats = cursor.fetchall()
    if chats:
        for chat in chats:
            print(f"  User: {chat[2][:50]}...")
            print(f"  Bot:  {chat[3][:50]}...")
            print()
    else:
        print("  (No chats yet)")
    
    # Stats
    print("\n📈 STATISTICS:")
    print("-" * 40)
    cursor.execute("SELECT COUNT(*) FROM users")
    print(f"  Total Users: {cursor.fetchone()[0]}")
    cursor.execute("SELECT COUNT(*) FROM diagnosis_history")
    print(f"  Total Scans: {cursor.fetchone()[0]}")
    cursor.execute("SELECT COUNT(*) FROM chat_history")
    print(f"  Total Chats: {cursor.fetchone()[0]}")
    
    conn.close()
    print("\n" + "=" * 60)

if __name__ == '__main__':
    view_all()
