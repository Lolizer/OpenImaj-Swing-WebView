import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.*;

import javax.imageio.ImageIO;
import javafx.application.Platform;
import javafx.embed.swing.JFXPanel;
import javafx.event.EventHandler;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.scene.web.WebEngine;
import javafx.scene.web.WebEvent;
import javafx.scene.web.WebView;

import javafx.util.Pair;
import netscape.javascript.JSObject;
import org.apache.commons.vfs2.FileSystemException;
import org.apache.http.HttpEntity;
import org.apache.http.HttpResponse;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.ContentType;
import org.apache.http.entity.mime.HttpMultipartMode;
import org.apache.http.entity.mime.MultipartEntityBuilder;
import org.apache.http.entity.mime.content.StringBody;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;

import org.openimaj.data.dataset.*;

import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.dataset.util.DatasetAdaptors;
import org.openimaj.feature.*;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.Transforms;
import org.openimaj.image.model.EigenImages;
import org.openimaj.image.processing.face.alignment.*;
import org.openimaj.image.processing.face.detection.DetectedFace;
import org.openimaj.image.processing.face.detection.FaceDetector;
import org.openimaj.image.processing.face.detection.HaarCascadeDetector;
import org.openimaj.image.processing.face.detection.keypoints.FKEFaceDetector;
import org.openimaj.image.processing.face.detection.keypoints.FacialKeypoint;
import org.openimaj.image.processing.face.detection.keypoints.KEDetectedFace;
import org.openimaj.image.processing.face.feature.LocalLBPHistogram;
import org.openimaj.image.processing.face.feature.comparison.FaceFVComparator;
import org.openimaj.image.processing.face.feature.comparison.FacialFeatureComparator;
import org.openimaj.image.processing.face.recognition.AnnotatorFaceRecogniser;
import org.openimaj.image.processing.face.recognition.FisherFaceRecogniser;
import org.openimaj.ml.annotation.ScoredAnnotation;
import org.openimaj.ml.annotation.basic.KNNAnnotator;

import javax.swing.*;
import javax.swing.table.DefaultTableModel;

import static javax.swing.WindowConstants.EXIT_ON_CLOSE;

public class JavFx {
    static WebView webView;

    int w;
    int h;
    int x;
    int y;

    volatile double maxVecVal;

    volatile BufferedImage bi;
    volatile Graphics2D g2d;
    FaceDetector<DetectedFace, FImage> fd = new HaarCascadeDetector(20);
    FaceDetector<KEDetectedFace, FImage> fkefd = new FKEFaceDetector(); // other FaceDetector with faces points
    FacialKeypoint[] fkp; //For FeDetector
    Map<String, DoubleFV[]> features;
    volatile EigenImages eigen;
    volatile DoubleFV testFeature;
    volatile Image res;
    volatile int rows = 150;
    volatile int columns = 150;

    BufferedImage bufferImage;
    BufferedImage bufferImage2;

    //For paintComponent's recognition operations
    volatile Image resAwt;
    volatile BufferedImage bufferImageAwt;
    volatile BufferedImage bufferImage2Awt;
    volatile Graphics2D resGAwt;
    volatile ArrayList<String> names; //recognized faces
    volatile ArrayList<String> namesAwt;
    volatile String bestPerson = null;
    volatile GroupedDataset<String, ListDataset<FImage>, FImage> training;

    GroupedDataset<String, ListDataset<KEDetectedFace>, KEDetectedFace> dataset;
    GroupedDataset<String, ListDataset<DetectedFace>, DetectedFace> KNN_dataset;


    AnnotatorFaceRecogniser<DetectedFace, String> recogniser;
    FisherFaceRecogniser<KEDetectedFace,String> fisher_recogniser;


    volatile FImage test;
    volatile Boolean isTrined = false;
    int amnt;
    Boolean checking = false;
    String thrName;

    byte method = 0;

    MBFImage[] mbf = new MBFImage[]{null, null};
    ArrayList<DetectedFace> faces;
    ArrayList<KEDetectedFace> kefaces; // for FKEFD ↑
    GraphicsConfiguration gConf = new JWindow().getGraphicsConfiguration(); //is used for image creating
    volatile Graphics2D g2D;
    BasicStroke bsk = new BasicStroke(5.0f);
    String buf;
    int width;
    int height;
    volatile boolean ifInside = false;
    volatile boolean ifInside2 = false;

    volatile boolean isSetRec = false;

    volatile JFXPanel fxPanel;

    static JavaBridge bridge2;

    //fields for specifying of directory names and the amount of pngs to be saved

    JTextField name = new JTextField(10);
    JTextField amount = new JTextField(10);


    Recognition rc;

    //turning on and off a long-running threads
    boolean flg = false;

    /* Create a JFrame with a JButton and a JFXPanel containing the WebView. */

    public JavFx() {
        JFrame frame = new JFrame("FX");
        Dimension dm = new Dimension(850, 630);

        frame.getContentPane().setLayout(null);

        final JTextArea url = new JTextArea(); //a field for entering an url addresses
        url.setToolTipText("Example: google.com");

        final JButton jButton = new JButton("Dataset");
        final JButton jLoad = new JButton("Load");

        JMenuBar jmb = new JMenuBar();
        JMenu jm = new JMenu("File");
        JMenuItem rec = new JMenuItem("Learn");

        JMenuItem setting = new JMenuItem("Settings");

        JPanel paramsPane = new JPanel();
        JPanel settingPane = new JPanel();
        settingPane.setLayout(new BoxLayout(settingPane,BoxLayout.Y_AXIS));

        final Checkbox ck = new Checkbox("Draw the facial rectangles within the window (Haar)");
        final Checkbox eigenCheck = new Checkbox("Eigenfaces recognition");
        final Checkbox AFRCheck = new Checkbox("AnnotatorFaceRecogniser KNN");
        final Checkbox fisherCheck = new Checkbox("Fisherfaces recognition + KEFaceDetector");

        final JTextField threshold = new JTextField(3);
        threshold.setToolTipText("Threshold for EigenFaces");
        threshold.setText("Threashold");

        //overriding of paintComponent() method for drawing operations in window
        fxPanel = new JFXPanel() {
            @Override
            public void paintComponent(Graphics g) {
                super.paintComponent(g);
                w = getWidth();
                h = getHeight();
                g2D = (Graphics2D) g;
                g2D.setPaint(Color.RED);
                g2D.setStroke(bsk);
                if (faces != null && ifInside) {

                    int bounderer = 0;

                    for (DetectedFace face : faces) {
                        if (face.getConfidence() < 8.0f) continue;
                        x = Math.round(face.getBounds().x);
                        y = Math.round(face.getBounds().y);
                        buf = "" + face.getConfidence();
                        width = Math.round(face.getBounds().width);
                        height = Math.round(face.getBounds().height);
                        g.drawString(buf, x + 10, y - 10);
                        g.drawRect(x, y, width, height);

                        thrName = Thread.currentThread().getName();

                        if (namesAwt != null && !thrName.equals("firstWorker") && !thrName.equals("secondWorker")) {

                            if (bounderer < namesAwt.size()) {
                                if (!namesAwt.get(bounderer).isEmpty()) {
                                    g.drawString(namesAwt.get(bounderer), x + 40, y - 10);
                                    bounderer++;
                                }
                            }
                            if (bounderer == namesAwt.size()) bounderer = 0;
                        }
                    }
                }
                if(kefaces != null){
                    int bounderer = 0;

                    for (KEDetectedFace face : kefaces) {
                        if (face.getConfidence() < 8.0f) continue;
                        x = Math.round(face.getBounds().x);
                        y = Math.round(face.getBounds().y);
                        buf = "" + face.getConfidence();
                        width = Math.round(face.getBounds().width);
                        height = Math.round(face.getBounds().height);
                        g.drawString(buf, x + 10, y - 10);
                        g.drawRect(x, y, width, height);

                        thrName = Thread.currentThread().getName();

                        if (namesAwt != null && !thrName.equals("firstWorker") && !thrName.equals("secondWorker")) {

                            if (bounderer < namesAwt.size()) {
                                if (!namesAwt.get(bounderer).isEmpty()) {
                                    g.drawString(namesAwt.get(bounderer), x + 40, y - 10);
                                    bounderer++;
                                }
                            }
                            if (bounderer == namesAwt.size()) bounderer = 0;
                        }
                    }
                }
            }
        };

        Platform.runLater(() -> { //runs long-running process for making images and facial detecting
            MixedProcess process = new MixedProcess();
            try {
                process.execute();
            } catch (Exception e) {
                e.printStackTrace();
            }
        });

        Platform.runLater(() -> { //runs second long-running process for making images and performing recognition
            LongRunProcess process2 = new LongRunProcess();
            try {
                process2.execute();
            } catch (Exception e) {
                e.printStackTrace();
                }
            }
        );

        //
        frame.addWindowListener(new WindowListener() {
            @Override
            public void windowOpened(WindowEvent e) {

            }

            @Override
            public void windowClosing(WindowEvent e) {
                Platform.runLater(new Runnable() { //closes frame consisting initFX and set the WebView to null
                    @Override
                    public void run() {
                        webView.getEngine().load(null);
                        frame.setVisible(false);
                    }
                });
            }

            @Override
            public void windowClosed(WindowEvent e) {

            }

            @Override
            public void windowIconified(WindowEvent e) {

            }

            @Override
            public void windowDeiconified(WindowEvent e) {

            }

            @Override
            public void windowActivated(WindowEvent e) {

            }

            @Override
            public void windowDeactivated(WindowEvent e) {

            }
        });

        //allows us to change a loaded page by pushing the jLoad button
        jLoad.addActionListener(e -> {
            Platform.runLater(() -> {
                if (!url.getText().equals(""))
                    webView.getEngine().load("http:\\\\" + url.getText());
            });
        });

        jButton.addActionListener(e -> { //Execute some JS code by pushing button
            Platform.runLater(() -> {
                if(!(JOptionPane.showConfirmDialog(null,"Are you sure?","Datasets handling",JOptionPane.YES_NO_OPTION,JOptionPane.QUESTION_MESSAGE) == JOptionPane.OK_OPTION))
                    return;
                if(dataset == null && KNN_dataset == null && training == null) {
                    JOptionPane.showMessageDialog(null, "Please train any algorithm first!", "Prompt", JOptionPane.WARNING_MESSAGE);
                    return;
                }
                if(dataset != null && method == 3)
                    if(!dataset.isEmpty()) {
                        frame.setEnabled(false);
                        JFrame jfr = new JFrame("Method: Fisherfaces + KEFaceDetection");
                        JLabel info = new JLabel("Training dataset:",SwingConstants.CENTER);
                        JButton erase = new JButton("Erase dataset");
                        DefaultTableModel model = new DefaultTableModel(new String[]{"Facial name","Image"},dataset.keySet().size()){
                            public Class getColumnClass(int column)
                            {
                                return getValueAt(0, column).getClass();
                            }
                        };

                        Iterator<String> it = dataset.getGroups().iterator();
                        for (int i = 0; i < dataset.keySet().size(); i++) {
                            String buff = it.next();
                            model.setValueAt(buff,i,0);
                            //model.setValueAt(dataset.getRandomInstance(buff),i,1);
                            model.setValueAt(new ImageIcon(ImageUtilities.createBufferedImage(dataset.getRandomInstance(buff).getFacePatch())),i,1);
                        }
                        JTable table = new JTable(model);
                        table.setPreferredScrollableViewportSize(table.getPreferredSize());
                        table.setCellSelectionEnabled(false);
                        table.setRowHeight(rows);
                        JScrollPane jsp = new JScrollPane(table);
                        jsp.setHorizontalScrollBarPolicy(JScrollPane.HORIZONTAL_SCROLLBAR_NEVER);
                        jsp.setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED);

                        erase.addActionListener((e2)->{
                            dataset = null;
                            fisherCheck.setState(false);
                            fisher_recogniser = null;
                            ck.setState(false);
                            kefaces = null;
                            faces = null;
                            names = null;
                            namesAwt = null;
                            model.setRowCount(0);
                            table.revalidate();
                            ifInside = false;
                        });

                        jfr.getContentPane().add(info,BorderLayout.NORTH);
                        jfr.getContentPane().add(jsp,BorderLayout.CENTER);
                        jfr.getContentPane().add(erase,BorderLayout.SOUTH);

                        jfr.setSize(400,rows + 200);

                        jfr.addWindowListener(new WindowListener() {
                            @Override
                            public void windowOpened(WindowEvent e) {

                            }

                            @Override
                            public void windowClosing(WindowEvent e) {
                                frame.setEnabled(true);
                            }

                            @Override
                            public void windowClosed(WindowEvent e) {

                            }

                            @Override
                            public void windowIconified(WindowEvent e) {

                            }

                            @Override
                            public void windowDeiconified(WindowEvent e) {

                            }

                            @Override
                            public void windowActivated(WindowEvent e) {

                            }

                            @Override
                            public void windowDeactivated(WindowEvent e) {

                            }
                        });

                        jfr.setVisible(true);
                    }

                if(KNN_dataset != null && method == 2)
                    if(!KNN_dataset.isEmpty()) {
                        frame.setEnabled(false);
                        JFrame jfr = new JFrame("Method: KNNAnnotator + HaarFaceDetection");
                        JLabel info = new JLabel("Training dataset:",SwingConstants.CENTER);
                        JButton erase = new JButton("Erase dataset");
                        DefaultTableModel model = new DefaultTableModel(new String[]{"Facial name","Image"},KNN_dataset.keySet().size()){
                            public Class getColumnClass(int column)
                            {
                                return getValueAt(0, column).getClass();
                            }
                        };

                        Iterator<String> it = KNN_dataset.getGroups().iterator();
                        for (int i = 0; i < KNN_dataset.keySet().size(); i++) {
                            String buff = it.next();
                            model.setValueAt(buff,i,0);
                            //model.setValueAt(dataset.getRandomInstance(buff),i,1);
                            model.setValueAt(new ImageIcon(ImageUtilities.createBufferedImage(KNN_dataset.getRandomInstance(buff).getFacePatch())),i,1);
                        }
                        JTable table = new JTable(model);
                        table.setPreferredScrollableViewportSize(table.getPreferredSize());
                        table.setCellSelectionEnabled(false);
                        table.setRowHeight(rows);
                        JScrollPane jsp = new JScrollPane(table);
                        jsp.setHorizontalScrollBarPolicy(JScrollPane.HORIZONTAL_SCROLLBAR_NEVER);
                        jsp.setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED);

                        erase.addActionListener((e2)->{
                            KNN_dataset = null;
                            AFRCheck.setState(false);
                            recogniser = null;
                            ck.setState(false);
                            faces = null;
                            names = null;
                            namesAwt = null;
                            model.setRowCount(0);
                            table.revalidate();
                            ifInside = false;
                        });

                        jfr.getContentPane().add(info,BorderLayout.NORTH);
                        jfr.getContentPane().add(jsp,BorderLayout.CENTER);
                        jfr.getContentPane().add(erase,BorderLayout.SOUTH);

                        jfr.setSize(400,rows + 200);

                        jfr.addWindowListener(new WindowListener() {
                            @Override
                            public void windowOpened(WindowEvent e) {

                            }

                            @Override
                            public void windowClosing(WindowEvent e) {
                                frame.setEnabled(true);
                            }

                            @Override
                            public void windowClosed(WindowEvent e) {

                            }

                            @Override
                            public void windowIconified(WindowEvent e) {

                            }

                            @Override
                            public void windowDeiconified(WindowEvent e) {

                            }

                            @Override
                            public void windowActivated(WindowEvent e) {

                            }

                            @Override
                            public void windowDeactivated(WindowEvent e) {

                            }
                        });

                        jfr.setVisible(true);
                    }

                if(training != null && method == 1)
                    if(!training.isEmpty()) {
                        frame.setEnabled(false);
                        JFrame jfr = new JFrame("Eigenfaces + HaarFaceDetection");
                        JLabel info = new JLabel("Training dataset:",SwingConstants.CENTER);
                        JButton erase = new JButton("Erase dataset");
                        DefaultTableModel model = new DefaultTableModel(new String[]{"Facial name","Image"},training.keySet().size()){
                            public Class getColumnClass(int column)
                            {
                                return getValueAt(0, column).getClass();
                            }
                        };

                        Iterator<String> it = training.getGroups().iterator();
                        for (int i = 0; i < training.keySet().size(); i++) {
                            String buff = it.next();
                            model.setValueAt(buff,i,0);
                            //model.setValueAt(dataset.getRandomInstance(buff),i,1);
                            model.setValueAt(new ImageIcon(ImageUtilities.createBufferedImage(training.getRandomInstance(buff))),i,1);
                        }
                        JTable table = new JTable(model);
                        table.setPreferredScrollableViewportSize(table.getPreferredSize());
                        table.setCellSelectionEnabled(false);
                        table.setRowHeight(rows);
                        JScrollPane jsp = new JScrollPane(table);
                        jsp.setHorizontalScrollBarPolicy(JScrollPane.HORIZONTAL_SCROLLBAR_NEVER);
                        jsp.setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED);

                        erase.addActionListener((e2)->{
                            training = null;
                            eigenCheck.setState(false);
                            eigen = null;
                            ck.setState(false);
                            faces = null;
                            names = null;
                            namesAwt = null;
                            model.setRowCount(0);
                            table.revalidate();
                            ifInside = false;
                        });

                        jfr.getContentPane().add(info,BorderLayout.NORTH);
                        jfr.getContentPane().add(jsp,BorderLayout.CENTER);
                        jfr.getContentPane().add(erase,BorderLayout.SOUTH);

                        jfr.setSize(400,rows + 200);

                        jfr.addWindowListener(new WindowListener() {
                            @Override
                            public void windowOpened(WindowEvent e) {

                            }

                            @Override
                            public void windowClosing(WindowEvent e) {
                                frame.setEnabled(true);
                            }

                            @Override
                            public void windowClosed(WindowEvent e) {

                            }

                            @Override
                            public void windowIconified(WindowEvent e) {

                            }

                            @Override
                            public void windowDeiconified(WindowEvent e) {

                            }

                            @Override
                            public void windowActivated(WindowEvent e) {

                            }

                            @Override
                            public void windowDeactivated(WindowEvent e) {

                            }
                        });

                        jfr.setVisible(true);
                    }
                /*
                CloseableHttpClient httpclient = HttpClients.createMinimal();
                HttpPost request = new HttpPost("http://bit.do/mod_perl/url-shortener.pl");
                StringBody action = new StringBody("shorten", ContentType.APPLICATION_FORM_URLENCODED);
                StringBody urlPost = new StringBody("http://opds.spbsut.ru/data/_uploaded/mu/vlss16mu_opentest.pdf", ContentType.APPLICATION_FORM_URLENCODED);
                StringBody url2Post = new StringBody(" site2 ", ContentType.APPLICATION_FORM_URLENCODED);
                StringBody url_hash = new StringBody("", ContentType.APPLICATION_FORM_URLENCODED);
                StringBody url_stats_is_private = new StringBody("0", ContentType.APPLICATION_FORM_URLENCODED);
                StringBody permission = new StringBody("1555799293|i7daeiv0bj", ContentType.APPLICATION_FORM_URLENCODED);

                MultipartEntityBuilder builder = MultipartEntityBuilder.create();
                builder.setMode(HttpMultipartMode.BROWSER_COMPATIBLE);
                builder.addPart("action", action);
                builder.addPart("url", urlPost);
                builder.addPart("url2", url2Post);
                builder.addPart("url_hash", url_hash);
                builder.addPart("url_stats_is_private", url_stats_is_private);
                builder.addPart("permasession", permission);

                HttpEntity entity = builder.build();

                request.setEntity(entity);
                try {
                    HttpResponse response = httpclient.execute(request);
                    HttpEntity he = response.getEntity();
                    InputStream in = new PushbackInputStream(he.getContent());

                    byte[] buff = in.readAllBytes();
                    if (buff.length == 0) System.out.println("SUCK!");

                    FileOutputStream fw = new FileOutputStream(new File("logger.txt"));
                    OutputStreamWriter osw = new OutputStreamWriter(fw, "UTF-8");
                    osw.write(response.toString());
                    osw.write("|-|");
                    fw.write(buff);

                    //webView.getEngine().load(EntityUtils.toString(he, "UTF-8")); //почему то не работает
                    osw.close();
                } catch (IOException ie) {
                    ie.printStackTrace();
                }

                //some another JS Script
                /*CloseableHttpClient httpclient = HttpClients.createMinimal();
                /*
                HttpClient client = new HttpClient();
                HttpMethod method = new PostMethod("https://cabinet.sut.ru/raspisanie_all_new?type_z=1");
                /
                HttpPost request = new HttpPost("https://cabinet.sut.ru/raspisanie_all_new?type_z=1");
                StringBody faculty = new StringBody("50029", ContentType.MULTIPART_FORM_DATA);
                StringBody kurs = new StringBody("4", ContentType.MULTIPART_FORM_DATA);
                StringBody group = new StringBody("52543", ContentType.MULTIPART_FORM_DATA);
                StringBody ok = new StringBody("Показать", ContentType.MULTIPART_FORM_DATA);
                StringBody group_el = new StringBody("0", ContentType.MULTIPART_FORM_DATA);

                MultipartEntityBuilder builder = MultipartEntityBuilder.create();
                builder.setMode(HttpMultipartMode.BROWSER_COMPATIBLE);
                builder.addPart("faculty",faculty);
                builder.addPart("kurs",kurs);
                builder.addPart("group",group);
                builder.addPart("ok",ok);
                builder.addPart("group_el",group_el);

                HttpEntity entity = builder.build();

                request.setEntity(entity);
                try {
                    HttpResponse response = httpclient.execute(request);
                    HttpEntity he = response.getEntity();
                    webView.getEngine().load(EntityUtils.toString(he, "UTF-8")); //почему то не работает
                }catch (IOException ie){
                    ie.printStackTrace();
                }
            }
            );
            */
            });
        });

        final JavaBridge bridge = new JavaBridge(); //allows us to get a console.log() method's results as System.out.println()

        bridge2 = bridge;

        ck.setState(false);
        ck.addItemListener(new ItemListener() {
            @Override
            public void itemStateChanged(ItemEvent e) {
                ifInside = !ifInside;
            }
        });

        eigenCheck.addItemListener(new ItemListener() { //prevents the use of several recognition methods concurrently
            @Override
            public void itemStateChanged(ItemEvent e) {
                if(!(dataset == null) || !(KNN_dataset == null)) {
                    JOptionPane.showMessageDialog(null, "You have to erase dataset of current algorithm", "Dataset issue", JOptionPane.WARNING_MESSAGE);
                    eigenCheck.setState(false);
                    return;
                }
                method = 1;
                AFRCheck.setState(false);
                fisherCheck.setState(false);
                if(eigenCheck.getState() == false) method = 0;
            }
        });

        AFRCheck.addItemListener(new ItemListener() { //does the same as previous one
            @Override
            public void itemStateChanged(ItemEvent e) {
                if(!(eigen == null) || !(dataset == null)) {
                    JOptionPane.showMessageDialog(null, "You have to erase dataset of current algorithm", "Dataset issue", JOptionPane.WARNING_MESSAGE);
                    AFRCheck.setState(false);
                    return;
                }
                method = 2;
                eigenCheck.setState(false);
                fisherCheck.setState(false);
                if(AFRCheck.getState() == false) method = 0;
            }
        });

        fisherCheck.addItemListener(new ItemListener() { //does the same as previous one
            @Override
            public void itemStateChanged(ItemEvent e) {
                if(!(eigen == null) || !(KNN_dataset == null)) {
                    JOptionPane.showMessageDialog(null, "You have to erase dataset of current algorithm", "Dataset issue", JOptionPane.WARNING_MESSAGE);
                    fisherCheck.setState(false);
                    return;
                }
                method = 3;
                eigenCheck.setState(false);
                AFRCheck.setState(false);
                if(fisherCheck.getState() == false) method = 0;
            }
        });

        fisherCheck.addItemListener((e)->{
            if(fisherCheck.getState() == true)
            JOptionPane.showMessageDialog(null,
            "Please make sure you set the amount greater than 20 for this method.\nRecognition won't be working until 3 faces get added to",
            "Warning", JOptionPane.WARNING_MESSAGE);
        });

        //setting of the component placements and their sizes
        jButton.setSize(new Dimension(200, 27));
        jButton.setLocation(0, 0);
        url.setBounds(Math.round(jButton.getWidth()) + 5, 5, 300,
                url.getPreferredSize().height);
        jLoad.setBounds(url.getX() + url.getWidth() + 5, 0, 70, jButton.getHeight());

        fxPanel.setSize(dm);
        fxPanel.setLocation(new Point(0, jButton.getHeight()));

        //adding of the menu on the frame
        jm.add(setting);
        jm.add(rec);
        jmb.add(jm);
        frame.setJMenuBar(jmb);

        //settings in MessageDialog
        setting.addActionListener((e)->{
            settingPane.add(threshold);
            settingPane.add(ck);
            settingPane.add(eigenCheck);
            settingPane.add(AFRCheck);
            settingPane.add(fisherCheck);

            JOptionPane.showMessageDialog(null, settingPane,"Settings",JOptionPane.OK_OPTION);
            if(threshold.getText().isEmpty() || threshold.getText().equals("Threashold")) {maxVecVal = 20; threshold.setText("" + maxVecVal); return;}
            //else threshold.setText(threshold.getText());
            try {
                maxVecVal = Math.round(Double.parseDouble(threshold.getText()));
            }catch (NumberFormatException nfe){
                threshold.setText("20.0");
                maxVecVal = 20;
                JOptionPane.showMessageDialog(null,"Threshold was set in the wrong way and so set by default");
            }
        });

        //starts learning operations for previously selected recognition method
        rec.addActionListener((e) -> {

            if(ck.getState() == false || (eigenCheck.getState() == false && AFRCheck.getState() == false && fisherCheck.getState() == false)){
                JOptionPane.showMessageDialog(null,"You have to first select method and turn detection on!","Warning",JOptionPane.WARNING_MESSAGE);
                return;
            }

            name.setText("Name:");
            if(amnt == 0) amount.setText("Amount:");

            paramsPane.add(name);

            if (!checking)
                for (Component cmp : paramsPane.getComponents())
                    if (cmp == amount) {
                        checking = true;
                        paramsPane.remove(amount);
                    }

            if(amount.getText().isEmpty() || amnt == 0) checking = false;

            if (!checking) paramsPane.add(amount);

            JOptionPane.showMessageDialog(null, paramsPane,"Parameters",JOptionPane.PLAIN_MESSAGE);
            if(!name.getText().isEmpty() && !amount.getText().isEmpty() && !name.getText().equals("Name:") && method != 0) {
                try{
                    Integer.parseInt(amount.getText());
                }catch(NumberFormatException nfe){
                    checking = false;
                    amount.setText("Amount:");
                    return;
                }
                rc = new Recognition();
                rc.execute();
            }

        });

        frame.add(jButton);
        frame.add(jLoad);
        frame.add(url);
        //frame.add(ck);
        frame.add(fxPanel);

        frame.setVisible(true);

        frame.getContentPane().setPreferredSize(dm);
        frame.pack();
        frame.setResizable(false);

        frame.setDefaultCloseOperation(EXIT_ON_CLOSE);

        Platform.runLater(new Runnable() { // this will run initFX as JavaFX-Thread
            @Override
            public void run() {
                initFX(fxPanel);
            }
        });
    }

    /* Creates a WebView */
    private static void initFX(final JFXPanel fxPanel) {
        Group group = new Group();
        Scene scene = new Scene(group);
        fxPanel.setScene(scene);

        webView = new WebView();

        group.getChildren().add(webView);
        webView.setMinSize(300, 300);
        webView.setPrefSize(850, 600);
        webView.setMaxSize(850, 600);

        webView.getEngine().setJavaScriptEnabled(true); //allows us to run JS code

        //START
        webView.getEngine().setOnAlert(new EventHandler<WebEvent<String>>() {
            @Override
            public void handle(WebEvent<String> event) {
                JOptionPane.showMessageDialog(
                        fxPanel,
                        event.getData(),
                        "Alert Message",
                        JOptionPane.ERROR_MESSAGE);
            }
        });

        webView.getEngine().getLoadWorker().stateProperty().addListener((observable, oldValue, newValue) ->
        {
            JSObject window = (JSObject) webView.getEngine().executeScript("window");
            window.setMember("java", bridge2);
            webView.getEngine().executeScript("console.log = function(message)\n" +
                    "{\n" +
                    "    java.log(message);\n" +
                    "};");
        });
        //END

        // Obtain the webEngine to navigate
        WebEngine webEngine = webView.getEngine();
        //solves Captcha problem
        //webEngine.setUserAgent("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36");
        webEngine.load("https://yandex.ru"); //sets the start page for the application
    }

    class MixedProcess extends SwingWorker {
        protected Object doInBackground() {
            Thread.currentThread().setName("firstWorker");
            while (true) {
                try {
                    if (w != 0 && h != 0 && w != 1 && h != 1 && !ifInside == false && flg == false) {
                        bi = gConf.createCompatibleImage(w, h);
                        g2d = bi.createGraphics();
                        fxPanel.paint(g2d);
                        g2d.dispose();
                        flg = !flg;
                        fxPanel.repaint();
                        fxPanel.validate();

                    }
                    if (bi != null && !ifInside == false && flg == true) {
                        mbf[0] = null;
                        faces = null;
                        kefaces = null;
                        mbf[0] = ImageUtilities.createMBFImage(bi, true);
                        if(method != 3) {
                            faces = (ArrayList<DetectedFace>) fd.detectFaces(Transforms.calculateIntensity(mbf[0]));
                        }else{
                            kefaces = (ArrayList<KEDetectedFace>) fkefd.detectFaces(Transforms.calculateIntensity(mbf[0]));
                        }
                        mbf[0] = null;
                        bi = null;
                        flg = !flg;
                        fxPanel.repaint();
                        fxPanel.validate();
                    }
                    // Sleep for a while
                    Thread.sleep(70);
                } catch (Exception e) {
                    e.printStackTrace();
                    return 1;
                }
            }
        }
    }

    class LongRunProcess extends SwingWorker {
        protected Object doInBackground() {
            Thread.currentThread().setName("secondWorker");
            while (true) {
                try {
                    if (!ifInside2) {
                        if (w != 0 && h != 0 && w != 1 && h != 1 && !ifInside == false) {
                            bi = gConf.createCompatibleImage(w, h);
                            g2d = bi.createGraphics();
                            fxPanel.paint(g2d);
                            g2d.dispose();
                        }
                        ifInside2 = !ifInside2;
                    } else {
                        if (faces != null && eigen != null && !features.isEmpty() && method == 1) {
                            int buffInd = eigen.toString().indexOf("dims=");
                            //if(eigen.getNumComponents() != 0)
                            if (!eigen.toString().substring(buffInd + 5, buffInd + 6).equals("0")) {
                                if (faces != null)
                                    if (!faces.isEmpty()) {
                                        //namesAwt = null;
                                        //System.out.println(eigen.toString().substring(eigen.toString().indexOf("dims=") + 5,eigen.toString().indexOf("dims=") + 6));
                                        names = new ArrayList<>(); // ВАЖНО! ПРИЧИНО ПО КОТОРОЙ ИМЕН НЕТ ИНОГДА
                                        for (DetectedFace face : faces) {
                                            if (face.getConfidence() < 8.0f) continue;
                                            bufferImageAwt = ImageUtilities.createBufferedImage(face.getFacePatch());
                                            resAwt = bufferImageAwt.getScaledInstance(columns, rows, Image.SCALE_SMOOTH);
                                            bufferImage2Awt = new BufferedImage(columns, rows, BufferedImage.TYPE_INT_ARGB);
                                            resGAwt = bufferImage2Awt.createGraphics();
                                            resGAwt.drawImage(resAwt, 0, 0, null);
                                            resGAwt.dispose();
                                            try {
                                                test = ImageUtilities.createFImage(bufferImage2Awt);
                                                //System.out.println(eigen + " END");
                                                //System.out.println(test.getBounds());
                                                testFeature = eigen.extractFeature(test);
                                            } catch (Exception e) {
                                                System.out.println(Thread.currentThread().getName() + "longprocess");
                                                e.printStackTrace();
                                                System.exit(1);
                                            }
                                            double minDistance = Double.MAX_VALUE;
                                            for (String pers : features.keySet()) {
                                                try {
                                                    if (!features.isEmpty()) {
                                                        //System.out.println(features.get(pers).length + "ENDING HERE BITCH");
                                                        DoubleFV[] dfv = features.get(pers);
                                                        //System.out.println(dfv);
                                                        for (final DoubleFV fv : dfv) {
                                                            double distance = fv.compare(testFeature, DoubleFVComparison.EUCLIDEAN);

                                                            if (distance < minDistance) {
                                                                minDistance = distance;
                                                                bestPerson = pers;
                                                            }
                                                        }
                                                    } else {
                                                        System.out.println("Features are null");
                                                    }
                                                } catch (Exception e) {
                                                    System.out.println(Thread.currentThread().getName() + "Longprocess2");
                                                    e.printStackTrace();
                                                    System.exit(1);
                                                }
                                            }

                                            //if (minDistance > maxVecVal * 0.8) names.add("Unknown");
                                            //else names.add(bestPerson);

                                            if (minDistance > maxVecVal) names.add("Unknown");
                                            else names.add(bestPerson);

                                            System.out.println(minDistance + " " + bestPerson);
                                        }
                                        namesAwt = names;
                                        System.out.println(names);
                                    }
                            }
                        }

                        ifInside2 = !ifInside2;

                        if(faces != null && method == 2 && recogniser != null && isSetRec){

                            namesAwt = null;
                            names = new ArrayList<>();

                            System.out.println();

                            for(DetectedFace face : faces){
                                ScoredAnnotation<String> entry = recogniser.annotateBest(face);
                                names.add(entry.annotation);
                                System.out.print("Annotated face (KNN): " + entry.annotation + " ");
                            }

                            namesAwt = names;
                            //System.out.println(namesAwt);
                        }

                        if(kefaces != null && method == 3 && fisher_recogniser != null && isSetRec){
                            if(!kefaces.isEmpty()) {
                                namesAwt = null;
                                names = new ArrayList<>();

                                System.out.println();

                                for (KEDetectedFace face : kefaces) {
                                    ScoredAnnotation<String> entry = fisher_recogniser.annotateBest(face);
                                        names.add(entry.annotation);
                                        System.out.print("Annotated fisherface: " + entry.annotation + " ");
                                }

                                namesAwt = names;
                            }
                        }

                        }
                    // Sleep for a while
                    Thread.sleep(100);
                } catch (RuntimeException re){
                    re.printStackTrace();
                    System.exit(3);
                } catch (Exception e) {
                    System.out.println(Thread.currentThread().getName() + "THAT BITCH2");
                    e.printStackTrace();
                    return 1;
                }
            }
        }
    }


    class Recognition extends SwingWorker {
        protected Object doInBackground() {
            int count = 0;
            String buff = "";
            File dir = new File("Images/");
            //dir.mkdir();
            File dir2 = new File(dir.getPath() + "/" + name.getText());
            //dir2.mkdir();
            File dir3 = new File("EigenImages/");
            //dir3.mkdir();

            if(method == 1) {dir.mkdir();dir2.mkdir();dir3.mkdir();}

            if (amnt == 0)
                amnt = Integer.parseInt(amount.getText());

            int nTraining = Math.round(amnt) - 2;

            if(method == 1) {
                try {
                    while (count < amnt) {
                        if (faces != null)
                            for (DetectedFace face : faces) {
                                if (face.getConfidence() < 8.0f) continue;
                                if (count < amnt) {
                                    bufferImage = ImageUtilities.createBufferedImage(face.getFacePatch());
                                    res = bufferImage.getScaledInstance(columns, rows, Image.SCALE_SMOOTH);
                                    bufferImage2 = new BufferedImage(columns, rows, BufferedImage.TYPE_INT_ARGB);
                                    resGAwt = bufferImage2.createGraphics();
                                    resGAwt.drawImage(res, 0, 0, null);
                                    resGAwt.dispose();
                                    buff = dir2.getAbsolutePath() + "/" + name.getText() + (Math.random() * 1000000 + ".png");
                                    ImageIO.write(bufferImage2, "png", new File(buff));
                                    count++;
                                }
                            }
                        else Thread.sleep(40);
                        Thread.sleep(200);
                    }
                } catch (NumberFormatException nfe) {
                    System.exit(1);
                } catch (Exception e) {
                    e.printStackTrace();
                    System.exit(2);
                }

                if (!buff.isEmpty())
                    try {
                        features = new HashMap<String, DoubleFV[]>();
                        int nEigenvectors = 1000;
                        VFSGroupDataset<FImage> dataset = new VFSGroupDataset<>(dir.getAbsolutePath(),
                                ImageUtilities.FIMAGE_READER);
                        //System.out.println(dataset);
                        //System.out.println(dataset.getGroups());

                        GroupedRandomSplitter<String, FImage> splitter = new GroupedRandomSplitter<String, FImage>(dataset, nTraining, 0, 5);
                        training = splitter.getTrainingDataset();
                        ArrayList<FImage> basisImages = (ArrayList<FImage>) DatasetAdaptors.asList(training);
                        isTrined = false;
                        eigen = null;
                        eigen = new EigenImages(nEigenvectors);
                        isTrined = true;
                        eigen.train(basisImages);

                        for (int i = 0; i < training.getGroups().size(); i++)
                            ImageIO.write(ImageUtilities.createBufferedImage(eigen.visualisePC(i + nTraining - 2)),
                                    "png", new File(dir3.getAbsolutePath() + "/" + /*ordering error*/ training.getGroups().toArray()[i] + "" + "_eigen.png"));

                        for (String pers : training.getGroups()) {
                            final DoubleFV[] fvs = new DoubleFV[nTraining];
                            for (int i = 0; i < nTraining; i++) {
                                final FImage faceHere = training.get(pers).get(i);
                                fvs[i] = eigen.extractFeature(faceHere);
                            }
                            features.put(pers, fvs);
                        }

                        /*
                        ArrayList<Double> vec = new ArrayList<>();
                        double midVal = 0;
                        for (String pers : features.keySet()) {
                            midVal = Integer.MAX_VALUE;
                            for (DoubleFV dfv : features.get(pers)) {
                                for (String persIns : features.keySet()) {
                                    for (DoubleFV dfvIns : features.get(persIns)) {
                                        double distance = 0;
                                        if (!dfvIns.equals(dfv)) {
                                            distance = dfvIns.compare(dfv, DoubleFVComparison.EUCLIDEAN);
                                            if (distance < midVal) {
                                                midVal = distance;
                                            }
                                        }
                                    }
                                }
                                vec.add(midVal);
                            }
                        }

                        maxVecVal = Collections.max(vec);
                        System.out.println(maxVecVal + "HERE IS VECTOR'S MAX VALUE. Threashold: " + 0.8 * maxVecVal);
                        */

                    } catch (FileSystemException fse) {
                        fse.printStackTrace();
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
            }

            if(method == 2) {
                try {
                    count = 0;
                    ArrayList<DetectedFace> lds = new ArrayList<DetectedFace>();
                    if(KNN_dataset == null) KNN_dataset = new MapBackedDataset<>();
                    final LocalLBPHistogram.Extractor<DetectedFace> extractor = new LocalLBPHistogram.Extractor<>(new IdentityAligner<DetectedFace>());
                    final FacialFeatureComparator<LocalLBPHistogram> comparator = new FaceFVComparator<LocalLBPHistogram, FloatFV>(FloatFVComparison.EUCLIDEAN);
                    final KNNAnnotator<DetectedFace, String, LocalLBPHistogram> knn = KNNAnnotator.create(extractor, comparator, 1);
                    isSetRec = false;
                    recogniser = null;
                    recogniser = AnnotatorFaceRecogniser.create(knn);
                    while (count < amnt) {
                        if (faces != null) {
                            if ((lds.size() + faces.size()) < amnt + 2) {
                                for (DetectedFace face : faces) {
                                    System.out.println("KNN - " + count);
                                    lds.add(face);
                                    count++;
                                }
                            } else if (lds.size() < amnt + 2) {
                                System.out.println("KNN - " + count);
                                Iterator<DetectedFace> it = faces.iterator();
                                for (int i = 0; (i < amnt - lds.size()) && it.hasNext() && count < amnt; i++) {
                                    lds.add(it.next());
                                    count++;
                                }
                            }
                            //recogniser.train();
                        } else Thread.sleep(40);

                        Thread.sleep(100);
                    }
                    KNN_dataset.put(name.getText(), new ListBackedDataset<DetectedFace>(lds));
                    recogniser.train(KNN_dataset);
                    isSetRec = true;
                    System.out.println(lds + "|+_+|\n" + KNN_dataset);
                }catch (Exception e) {
                    e.printStackTrace();
                }
            }

            if(method == 3){
                try {
                    count = 0;
                    ArrayList<KEDetectedFace> lds = new ArrayList<KEDetectedFace>();
                    if(dataset == null) dataset = new MapBackedDataset<>();

                    isSetRec = false;
                    fisher_recogniser = FisherFaceRecogniser.create(30, new RotateScaleAligner(),1,DoubleFVComparison.EUCLIDEAN);
                    while (count < amnt) {
                        if (kefaces != null) {
                            if ((lds.size() + kefaces.size()) < amnt + 2) {
                                for (KEDetectedFace face : kefaces) {
                                    System.out.println("Fisher - " + count);
                                    lds.add(face);
                                    count++;
                                }
                            } else if (lds.size() < amnt + 2) {
                                System.out.println("Fisher - " + count);
                                Iterator<KEDetectedFace> it = kefaces.iterator();
                                for (int i = 0; (i < amnt - lds.size()) && it.hasNext() && count < amnt; i++) {
                                    lds.add(it.next());
                                    count++;
                                }
                            }
                            //recogniser.train();
                        } else Thread.sleep(40);

                        Thread.sleep(100);
                    }
                    dataset.put(name.getText(), new ListBackedDataset<KEDetectedFace>(lds)); //adds new face

                    for(String entry : dataset.keySet())
                        System.out.println("Faces we have: " + entry + " " + dataset.getInstances(entry));
                    if(dataset.size() > 2){
                        fisher_recogniser.train(dataset);
                        isSetRec = true;
                    }
                    //System.out.println(lds + "|+_+|\n" + dataset);

                }catch (Exception e) {
                    e.printStackTrace();
                }
            }

            return 0;
        }
    }


    public class JavaBridge {
        public void log(String text) {
            System.out.println(text);
        }
    }

}
