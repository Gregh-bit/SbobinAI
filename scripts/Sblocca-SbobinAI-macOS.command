#!/bin/bash
clear
echo "=========================================================="
echo "  SbobinAI - Sblocco Sicurezza Gatekeeper (macOS)"
echo "=========================================================="
echo ""
echo "Rimozione dell'attributo di quarantena da /Applications/SbobinAI.app..."

if [ -d "/Applications/SbobinAI.app" ]; then
    xattr -cr /Applications/SbobinAI.app 2>/dev/null
    echo ""
    echo "✅ Sblocco completato con successo!"
    echo "Avvio di SbobinAI in corso..."
    open /Applications/SbobinAI.app
else
    echo ""
    echo "⚠️ SbobinAI.app non è stata trovata nella cartella Applicazioni."
    echo "Assicurati di aver aperto il file .dmg e aver trascinato SbobinAI in Applicazioni prima di eseguire questo script."
fi

echo ""
read -p "Premi [Invio] per chiudere questa finestra..."

