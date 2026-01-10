import { StatusBar } from 'expo-status-bar';
import { StyleSheet, SafeAreaView, Text, View } from 'react-native';
import { WebView } from 'react-native-webview';

// =========================================================================
// IMPORTANT - REPLACE THIS WITH YOUR COMPUTER'S LOCAL IP ADDRESS
// 
// 1. Open Terminal/Command Prompt on your computer
// 2. Type 'ipconfig' (Windows) or 'ifconfig' (Mac/Linux)
// 3. Look for 'IPv4 Address' (usually looks like 192.168.1.xxx)
// 4. Replace the IP below with yours. 
//
// DO NOT use 'localhost' or '127.0.0.1' - that refers to the Phone itself!
// =========================================================================
const FLASK_SERVER_URL = 'https://crop-disease-detection-app-un09.onrender.com';

export default function App() {
    return (
        <SafeAreaView style={styles.container}>
            <StatusBar style="auto" />
            <WebView
                source={{ uri: FLASK_SERVER_URL }}
                style={{ flex: 1 }}
                javaScriptEnabled={true}
                domStorageEnabled={true}
                startInLoadingState={true}
                scalesPageToFit={true}
                onError={(syntheticEvent) => {
                    const { nativeEvent } = syntheticEvent;
                    console.warn('WebView error: ', nativeEvent);
                }}
                renderError={(errorName) => (
                    <View style={styles.errorContainer}>
                        <Text style={styles.errorText}>Unable to connect to:</Text>
                        <Text style={styles.urlText}>{FLASK_SERVER_URL}</Text>
                        <Text style={styles.helpText}>Make sure your computer and mobile are on the same WiFi.</Text>
                        <Text style={styles.helpText}>Make sure 'python app.py' is running.</Text>
                        <Text style={styles.helpText}>Check if the IP inside App.js is correct.</Text>
                    </View>
                )}
            />
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#fff',
    },
    errorContainer: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        padding: 20,
    },
    errorText: {
        fontSize: 18,
        color: 'red',
        marginBottom: 10,
    },
    urlText: {
        fontSize: 16,
        marginBottom: 20,
        fontWeight: 'bold',
    },
    helpText: {
        fontSize: 14,
        color: '#666',
        marginBottom: 5,
        textAlign: 'center',
    }
});
