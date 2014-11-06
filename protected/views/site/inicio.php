<style>
    #block_l4_headers,#block_l5_logos,#breadcrumbs_bar
    {
        display: none;
    }

    .operaciones,#h_sidebar
    {
        display: none;
    }
    #block_super_headers{
        display:inline-block;
    }
    .logoIzq
    {  
        min-width: 100px;
        width:30%;
        float:left;
    }
    .logoDer
    {
        min-width: 100px;
        width:30%;
        float:right;
        display: inline-block;
    }
    .ui-autocomplete{
        text-align:left;
    }
    .boton_b
    {
        width: 150px;
        background-color:#D6DCE8;opacity: 0.70 ;
        border:#c7cccf 1px solid;
    box-shadow: 2px 2px 2px rgba(0, 0, 0, 0.256875);
    -moz-box-shadow:2px 2px 2px rgba(0, 0, 0, 0.256875);
    -webkit-box-shadow:2px 2px 2px rgba(0, 0, 0, 0.256875);
    -moz-border-radius:8px;
    -webkit-border-radius: 8px;
    border-radius:8px;
        
    }
</style>


<?php
//$this->layout = 'column1';
// $this->layout = '//layouts/responsiveLayout';
?>


<?php $this->pageTitle = Yii::app()->name; ?>



<?php
$this->breadcrumbs = array(
    'Inicio',
);

//echo '<p>'.$browser.'</p>';
/* Detecta la versión de IE y selecciona el layout más adecuado. */
?>

<!--[if lt IE 9]>
    <div class="forms100cb" style="background-color:#FED;border: 2px solid #FA0;"><b style="color:#F00;">ATENCIÓN: </b><b>Se ha detectado que su navegador Web no soporta HTML5/CSS3 y puede ser inseguro.</b><br/>Haga click <a href="/chrome/GoogleChromeframeStandaloneEnterprise.msi">aquí</a> para instalar "<a href="/chrome/GoogleChromeframeStandaloneEnterprise.msi">Google Chrome Frame</a>" que <b>no requiere permisos de administrador</b> o actualice su navegador. Puede usar la aplicación pero experimentara bajo rendimiento.<br/></div>
    <![endif]-->
<?php
Yii::app()->clientScript->registerScript('highlightAC', '$.ui.autocomplete.prototype._renderItem = function (ul, item) {
  item.label = item.label.replace(new RegExp("(?![^&;]+;)(?!<[^<>]*)(" + $.ui.autocomplete.escapeRegex(this.term) + ")(?![^<>]*>)(?![^&;]+;)", "gi"), "<strong>$1</strong>");
  return $("<li></li>")
  .data("item.autocomplete", item)
  .append("<a>" + item.label + "</a>")
  .appendTo(ul);
  };', CClientScript::POS_END);
?>
<?php $this->pageTitle = 'CBM - Inicio' ?>
<div id="h_Arbol" class="h_Arbol">
    <p>Por favor, seleccione la ubicación usando el árbol y haga click en una operación o busque por TAG, código SAP o nombre.</p>

    <?php echo CHtml::form("", 'get', array()); ?>
    <div class="forms100c" style="text-align:left;">
        <?php
// Si existe el parámetro $_GET[OT]
// lee valores de get desde
        $model = new Motores();
        $modeloValue = '';
        if (isset($_GET['query'])) {
            $modeloValue = $_GET['query'];
        }
        $this->widget('zii.widgets.jui.CJuiAutoComplete', array(
            'name' => 'query',
            'source' => CController::createUrl('motores/tituloSearch'),
            'options' => array(
                'minLength' => '1',
            //'select' => 'js:function(event, ui) { console.log(ui.item.id +":"+ui.item.value); }',
            ),
            'htmlOptions' => array(
                'style' => 'width:50%; text-align:left;'
            ),
            'model' => $model,
            'value' => $modeloValue,
        ));

        if (isset($_GET['query'])) {
            //  echo '<div class="forms100c" style="padding-left:15px;padding-bottom:15px;width:600px">';
            echo '<legend>Resultados de la búsqueda</legend>';
            $contador = 0;
            $rutas = array();
            // Para Motores
            $modelos = Motores::model()->findAllBySql(
                    'SELECT * FROM motores WHERE 
                    (TAG LIKE "%' . $_GET['query'] . '%" OR Motor LIKE "%' . $_GET['query'] . '%")
            ');
            // para cada resultado del query Tipo 0=motor, 1=tablero, 2 =equipo
            foreach ($modelos as $modelo) {
                $modulos="  <a class=boton_b href=/index.php/aceitesnivel1/admin?id=".$modelo->id.">Lubricación</a>
                            <a class=boton_b href=/index.php/termomotores/admin?id=".$modelo->id.">Termografía</a>
                            <a class=boton_b href=/index.php/vibraciones/admin?id=".$modelo->id.">Vibraciones y T°</a>
                            <a class=boton_b href=/index.php/aislamiento_tierra/admin?id=".$modelo->id.">Aislamiento</a>
                            <a class=boton_b href=/index.php/motores/".$modelo->id.">Detalles</a>
                        ";
                array_push($rutas, array('TAG' => $modelo->TAG, 'nombre' => $modelo->Motor, 'tipo' => 'Motor','id_mod'=>$modelo->id, 'id' => $contador++,
                        'modulos'=>$modulos,
                    ));
            }
            // Para Tableros
            $modelos = Tableros::model()->findAllBySql(
                    'SELECT * FROM tableros WHERE 
                    (TAG LIKE "%' . $_GET['query'] . '%" OR Tablero LIKE "%' . $_GET['query'] . '%")
            ');
            // para cada resultado del query Tipo 0=motor, 1=tablero, 2 =equipo
            foreach ($modelos as $modelo) {
                $modulos="  <a href=/index.php/termotablero/admin?id=".$modelo->id.">Termografía</a>
                            <a href=/index.php/motores/".$modelo->id.">Detalles</a>
                        ";
                array_push($rutas, array('TAG' => $modelo->TAG, 'nombre' => $modelo->Tablero, 'tipo' => 'Tablero','id_mod'=>$modelo->id, 'id' => $contador++,
                        'modulos'=>$modulos,
                    ));
            }
             // Para Equipos
            $modelos = Estructura::model()->findAllBySql(
                    'SELECT * FROM estructura WHERE 
                    (Codigo LIKE "%' . $_GET['query'] . '%" OR Equipo LIKE "%' . $_GET['query'] . '%")
            ');
            // para cada resultado del query Tipo 0=motor, 1=tablero, 2 =equipo
            foreach ($modelos as $modelo) {
                $modulos="  <a href=/index.php/reportes/admin?id=".$modelo->id.">Ultrasonido</a>
                            <a href=/index.php/motores/admin?id=".$modelo->id.">Motores</a>
                            <a href=/index.php/estructura/".$modelo->id.">Detalles</a>
                        ";
                array_push($rutas, array('TAG' => $modelo->Codigo, 'nombre' => $modelo->Equipo, 'tipo' => 'Equipo','id_mod'=>$modelo->id, 'id' => $contador++,
                        'modulos'=>$modulos,
                    ));
            }           
        $dataProvider = new CArrayDataProvider($rutas, array(
                    'id' => 'user',
                    'pagination' => array(
                        'pageSize' => 20,
                    ),
                ));

        ?>

        <?php
        
        $this->widget('zii.widgets.grid.CGridView', array(
            'id' => 'buscar-grid',
            'dataProvider' => $dataProvider,
            // 'filter' => $model,
            'cssFile' => '/themes/gridview/styles.css', 'template' => '{items}{pager}{summary}', 'summaryText' => 'Resultados del {start} al {end} de {count} encontrados',
            'columns' => array(
                // 'titulo',
                array(// related city displayed as a link
                    'header' => 'Tipo',
                    'type' => 'raw',
                    'value' => 'isset($data["tipo"])?$data["tipo"]:""',
                ),
                array(// related city displayed as a link
                    'header' => 'TAG',
                    'type' => 'raw',
                    'value' => 'isset($data["TAG"])?$data["TAG"]:""',
                ),
                array(// related city displayed as a link
                    'header' => 'Nombre',
                    'type' => 'raw',
                    'value' => 'isset($data["nombre"])?$data["nombre"]:""',
                ),
                                array(// related city displayed as a link
                    'header' => 'Módulos',
                    'type' => 'raw',
                    'value' => 'isset($data["modulos"])?$data["modulos"]:""
                        
                        ',
                ),
            ),
        ));
        }
        //echo CHtml::textField('query', isset($_GET['query']) ? $_GET['query'] : "", array("style"=>"width:300px;"));
        ?>
  
    <div class="row buttons">
        <?php echo CHtml::submitButton(Yii::t('app', 'Aceptar')); ?>
        <a href='/index.php/metaDocs/admin?avanzada=1' style="margin-left:10px;">Búsqueda Avanzada</a>
    </div>
        <?php
//crea el formulario
        echo CHtml::endForm();
        ?>






<?php

//funció h_encode() coloca la letra H al inicio y codifica los espacios como : y los + como :ZZ
function h_encode($entrada) {
    $salida = "H__" . $entrada;
    $salida = str_replace(" ", "-__", $salida);
    $salida = str_replace("+", "-ZZ", $salida);
    $salida = str_replace("Ãƒï¿½", "-AA", $salida);
    $salida = str_replace("Ãƒâ€°", "-EE", $salida);
    $salida = str_replace("Ãƒï¿½", "-II", $salida);
    $salida = str_replace("Ãƒâ€œ", "-OO", $salida);
    $salida = str_replace("ÃƒÅ¡", "-UU", $salida);
    $salida = str_replace("Ãƒâ€˜", "-NN", $salida);
    return($salida);
}

function h_decode($entrada) {
    $salida = str_replace("H__", "", $entrada);
    $salida = str_replace("-__", " ", $salida);
    $salida = str_replace("-ZZ", "+", $salida);
    $salida = str_replace("-AA", "Ãƒï¿½", $salida);
    $salida = str_replace("-EE", "Ãƒâ€°", $salida);
    $salida = str_replace("-II", "Ãƒï¿½", $salida);
    $salida = str_replace("-OO", "Ãƒâ€œ", $salida);
    $salida = str_replace("-UU", "ÃƒÅ¡", $salida);
    $salida = str_replace("-NN", "Ãƒâ€˜", $salida);
    return($salida);
}
?>
    <?php

// CampoArea() Esta función Imprime el enlace y las opciones para cada Ãƒï¿½rea, el
// Parámetros:
// cArea = Nombre del area


    function CampoArea($cArea) {
        // inicializa cadena vacía
        $cadSalida = "";
        // Adiciona Título 
        $cadSalida = $cadSalida . CHtml::link($cArea->Proceso
                        . CHtml::link("&nbsp;", "#", array('class' => 'cAreaVacio',
                            'onclick' => '$("#miTree2").jstree("toggle_node", "#' . $cArea->Proceso . '");',
                        ))
                        . CHtml::link('Lubricantes', '/index.php/tipo/admin?area=' . $cArea->Proceso, array(
                            'onclick' => 'location.href="/index.php/tipo/admin?area=' . $cArea->Proceso . '"',
                            'class' => 'cArea'
                        ))
                        . CHtml::link('Resultados', '/index.php/site/page?view=resumen&area=' . $cArea->Proceso, array(
                            'onclick' => 'location.href="/index.php/site/page?view=resumen&area=' . $cArea->Proceso . '"',
                            'class' => 'cArea'
                        ))
                        , "#", array('class' => 'cAreaNombre',
                    'onclick' => '$("#miTree2").jstree("toggle_node", "#' . $cArea->Proceso . '");',
                ));
        // Adiciona línea horizontal de abajo
        //$cadSalida=$cadSalida.$boton = CHtml::linkButton('Ver', array('submit' => '/index.php/motores/admin'));
        return($cadSalida);
    }

// CampoProceso() Esta función Imprime el enlace y las opciones para cada Proceso, el
// Parámetros:
// cProceso = Nombre del proceso
    function CampoProceso($cProceso) {
        $cadSalida = "";
        $cadSalida = $cadSalida . CHtml::link($cProceso->Area, "#", array('class' => 'cProcesoNombre',
                    'onclick' => '$("#miTree2").jstree("toggle_node", "#' . h_encode($cProceso->Area) . '");'
                ))
                . CHtml::link("&nbsp;", "#", array('class' => 'cProcesoVacio',
                    'onclick' => '$("#miTree2").jstree("toggle_node", "#' . h_encode($cProceso->Area) . '");',
                ))
                . CHtml::link('Estr. Tableros', '/index.php/tableros/admin?proceso=' . $cProceso->Area, array(
                    'onclick' => 'location.href="/index.php/tableros/admin?proceso=' . $cProceso->Area . '"',
                    'class' => 'cProceso'
                ))
                . CHtml::link('Equipos', '/index.php/estructura/admin?proceso=' . $cProceso->Area, array(
                    'onclick' => 'location.href="/index.php/estructura/admin?proceso=' . $cProceso->Area . '"',
                    'class' => 'cProceso'
                ));
        //D6DCE8
        // Adiciona línea horizontal de abajo
        //$cadSalida=$cadSalida.$boton = CHtml::linkButton('Ver', array('submit' => '/index.php/motores/admin'));
        return($cadSalida);
    }

//B2D67C
// coloca el datatree vacío.        
    $dataTree = "";
// encuentra todas las áreas (columna proceso)
    $areas = Estructura::model()->findAllBySql("SELECT DISTINCT Proceso FROM estructura ORDER BY Proceso");
// para cada área adiciona una entrada y sus acciones, 
    $na = 0;
    foreach ($areas as $area) {
        $dataTree = $dataTree . '<li class="jstree-open" id="' . $area->Proceso . '">'; // Area
        $dataTree = $dataTree . CampoArea($area);
        $dataTree = $dataTree . "<ul>"; // Area
        $procesos = Estructura::model()->findAllBySql('SELECT DISTINCT Area FROM estructura WHERE Proceso="' . $area->Proceso . '" ORDER BY Area');
        // para cada proceso
        $np = 0;
        foreach ($procesos as $proceso) {
            $dataTree = $dataTree . '<li id="' . h_encode($proceso->Area) . '">'; // Proceso
            $dataTree = $dataTree . CampoProceso($proceso);
            $dataTree = $dataTree . "<ul>"; // Proceso
            //$dataTree = $dataTree . '<li id="' . $proceso->Proceso. '" class="jstree-closed" style=font-size=13px !important;"">'; // Letrero de Tablero
            $dataTree = $dataTree . '<li id="' . h_encode($proceso->Area) . '_tableros' . '" tipo="Tablero" class="jstree-closed" style=font-size=13px !important;"">'; // Letrero de Tablero
            //$dataTree = $dataTree . '<a href="#" style="width: 97%;" onclick="$(\"#miTree2\").jstree(\"toggle_node\", \"#'.h_encode($proceso->Area).'_tableros");'.'>Tableros</a>';
            $dataTree = $dataTree . CHtml::link("Tableros", "#", array('style' => 'width: 96%;',
                        'onclick' => '$("#miTree2").jstree("toggle_node", "#' . h_encode($proceso->Area) . '_tableros");'
                    ));
            $dataTree = $dataTree . "<ul>"; // Tableros
            /* Para AINIC Tableros        // para cada tablero
              $tableros = Tableros::model()->findAllBySql('SELECT Tablero,id,TAG FROM tableros WHERE Area="' . $proceso->Area . '" ORDER BY Tablero');
              foreach ($tableros as $tablero) {
              $dataTree=$dataTree."<li>"; // Tablero
              $dataTree=$dataTree.CampoTablero($tablero);
              $dataTree=$dataTree."</li>"; // Tablero
              }
             */
            $dataTree = $dataTree . "</ul>";  // Tableros
            $dataTree = $dataTree . "</li>"; // Letrero de Equipos
            $dataTree = $dataTree . '<li id="' . h_encode($proceso->Area) . '_equipos' . '" tipo="Equipos" class="jstree-closed" style=font-size=13px !important;"">'; // Letrero de Equipos
            //$dataTree = $dataTree . '<a href="#" style="width: 97%;">Equipos</a>';
            $dataTree = $dataTree . CHtml::link("Equipos", "#", array('style' => 'width: 96%;',
                        'onclick' => '$("#miTree2").jstree("toggle_node", "#' . h_encode($proceso->Area) . '_equipos");'
                    ));
            $dataTree = $dataTree . "<ul>"; // Equipos
            /* Para INIC Equipos
              $equipos = Estructura::model()->findAllBySql('SELECT Equipo,id,Codigo FROM estructura WHERE Area="' . $proceso->Area . '" ORDER BY Equipo');
              // para cada proceso
              $ne = 0;
              foreach ($equipos as $equipo) {
              $dataTree=$dataTree."<li>"; // Equipo
              $dataTree=$dataTree.CampoEquipo($equipo);
              $dataTree=$dataTree."<ul>"; // Equipo
              $motores = Motores::model()->findAllBySql('SELECT Motor, id,TAG FROM motores WHERE Equipo="' . $equipo->Equipo . '" ORDER BY Motor');
              foreach ($motores as $motor) {
              $dataTree=$dataTree."<li>"; // Motor
              $dataTree=$dataTree.CampoMotor($motor);
              $dataTree=$dataTree."</li>"; // Motor
              }
              $ne++;
              $dataTree=$dataTree."</ul>"; // Equipo
              $dataTree=$dataTree."</li>"; // Equipo
              }
             */
            $dataTree = $dataTree . "</ul>"; //  Equipos
            $dataTree = $dataTree . "</li>"; // Letrero de Equipos
            $np++;
            $dataTree = $dataTree . "</ul>"; //Proceso
            $dataTree = $dataTree . "</li>"; //Proceso
        }
        $na++;
        $dataTree = $dataTree . "</ul>"; //Area
        $dataTree = $dataTree . "</li>"; //Area
    }

// echo $dataTree.'<br /><br />';
    ?>
    <form>
        <input type="hidden" id="proceso" value="OK"/>
    </form>
    <script type="text/_javascript_" src=""></script>
    <style>
        fieldset {
            border:#e0cd7c 1px solid;
            -moz-border-radius:3px;
            -webkit-border-radius: 3px;
            border-radius:3px;
        }

    </style>
    <div class="block_3d">
        <fieldset 
            style='
            border:#e0cd7c 1px solid;
            -moz-border-radius:8px;
            -webkit-border-radius: 8px;
            border-radius:8px;
            padding: 6px; margin:0px;
            width:98%;font-size:13px;
            '>

            <!--
            <input type="button" value="Collapse All" onclick='//$("#miTree2").jstree("close_all");
                                                        //var node = $("#miTree2").jstree("get_node",this.data.ui.hovered);
                                                      $("#miTree2").jstree("toggle_node","FILTRACION");
                                                                                               '>
            <input type="button" value="Expand All" onclick="$('#miTree2').jstree('open_all');">    
            
            -->

<?php
echo '<div class="hTest"><b slyle="text-color:#555555;">CV1 - Cervecería del Valle</b></div>';

$this->widget("application.extensions.jstree.JSTree", array(
    "name" => "miTreeview",
    "id" => "miTreeview",
    'bind' => array(
        "click.jstree" => 'function (event,data) {
                   var node = $(event.target).closest("li");
                   //alert(data);
                   $("#js_tree_jsTreeDefaultAttribute div").jstree("toggle_node",data);
                   return false;
            }
        '

//function(n, t) {t.toggle_branch(n);}",
    ),
    "plugins" => array(
        "html_data" => array(
            "data" => $dataTree,
            "ajax" => array(
                "data" => 'js:function(n)
                    {
                        return{"id" : n.attr ? n.attr("id") : 1, "tipo" : n.attr ? n.attr("tipo") : 1};
                    }
                ',
                "url" => CController::createUrl('/site/getArbol'),
            )
        ),
        //"sort",
        "ui" => array('select_limit' => 0), // para que no se use multi-select
        "themes" => array(
            "theme" => "tree",
            "url" => "/themes/tree/style.css",
            "icons" => false,
        ),
        /*
          "types" => array (
          "types" => array (
          "default" => array(
          "select_node"=>'function (e) {
          this.toggle_node(e);
          return false;
          }',
          ),
          ),
          ),
         * */

        "hotkeys" => array(
            "enabled" => true,
            "space" => '
                    var node = this._get_node(this.data.ui.hovered);
                    $("#miTree2").jstree("toggle_node",node);
                '
        ),
    ),
));
?>

        </fieldset>
    </div>
</div>

