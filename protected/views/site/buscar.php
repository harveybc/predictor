<style>
    .ui-corner-all{
        z-index:100;
    }
</style>
   <?php
    Yii::app()->clientScript->registerScript('highlightAC', '$.ui.autocomplete.prototype._renderItem = function (ul, item) {
  item.label = item.label.replace(new RegExp("(?![^&;]+;)(?!<[^<>]*)(" + $.ui.autocomplete.escapeRegex(this.term) + ")(?![^<>]*>)(?![^&;]+;)", "gi"), "<strong>$1</strong>");
  return $("<li></li>")
  .data("item.autocomplete", item)
  .append("<a>" + item.label + "</a>")
  .appendTo(ul);
  };', CClientScript::POS_END);
    ?><?php
$this->pageTitle = Yii::app()->name . ' - Index';
$this->breadcrumbs = array(
    'Index',
);
$sufix = "";
if (isset($_GET['query'])) {
        $sufix = "?query=" . urlencode($_GET['query']);
}
?>

<?php
// encuentra el modelo del usuario actual Yii::app()->user->name
//$modelo_user = Usuarios::model()->findByAttributes(array('Username' => Yii::app()->user->name));
//if (isset($modelo_user)) {
//    if ($modelo_user->esAdministrador) {
// TODO: Limitar este menú a usuarios autenticados
        $this->menu = array(
            array('label' => 'Subir Doc. de Motor', 'url' => array('/Documentos/createSubirMotor'.$sufix)),
            array('label' => 'Subir Doc. de Equipo', 'url' => array('/Documentos/createSubirEquipo'.$sufix)),
            array('label' => 'Subir Doc. de Tablero', 'url' => array('/Documentos/createSubirTablero'.$sufix)),
        );
 //   }
//}
?>  



<?php $this->setPageTitle ('Búsqueda de Documentos'); ?>
<div class="form">
    <b></b>
    Por favor introduzca un término, tag, código SAP relacionado o palabras clave.
    <br/>
    <?php echo CHtml::form("", 'get', array()); ?>

        <?php
// Si existe el parámetro $_GET[OT]
            
// lee valores de get desde
    $model=new MetaDocs();
    $modeloValue = '';
            if (isset($_GET['query'])) {
                $modeloValue=$_GET['query'];
            }
            $this->widget('zii.widgets.jui.CJuiAutoComplete', array(
                'name' => 'query',
                'source' => CController::createUrl('metaDocs/tituloSearch'),
                'options' => array(
                    'minLength' => '1',
                    'select' => 'js:function(event, ui) { console.log(ui.item.id +":"+ui.item.value); }',
                ),
                'htmlOptions' => array(
                    'style' => 'width:400px;'
                ),
                'model' => $model,
                'value' => $modeloValue,
            ));
                
    
    //echo CHtml::textField('query', isset($_GET['query']) ? $_GET['query'] : "", array("style"=>"width:300px;"));
    ?>
    
    <?php
// Si existe el parámetro $_GET[OT]

//    echo CHtml::textField('query', isset($_GET['query']) ? $_GET['query'] : "", array("style"=>"width:300px;"));

    ?>
    <div class="row buttons">
        <?php echo CHtml::submitButton(Yii::t('app', 'Aceptar')); ?>
    </div>
    <?php
//crea el formulario
    echo CHtml::endForm();
    ?>

    <?php
    // si existe el parámetro QUERY
    if (isset($_GET['query'])) {
        //  echo '<fieldset style="padding-left:15px;padding-bottom:15px;width:600px">';
        echo '<legend>Resultados de la búsqueda</legend>';
        $contador = 0;
        $rutas = array();
        // busca en cada modelo la OT

        $modelos = MetaDocs::model()->findAllBySql(
                'SELECT id,numPedido,titulo FROM metaDocs WHERE 
                    (numPedido LIKE "%' . $_GET['query'] . '%")
        ');
        // para cada resultado
        foreach ($modelos as $modelo) {
            array_push($rutas, array('titulo' => $modelo->titulo, 'ruta' => '/index.php/metaDocs/view/' . $modelo->id, 'ubicacion' => 'Número de Pedido', 'contenido' => $modelo->numPedido, 'id' => $contador++));
        }

        $modelos = MetaDocs::model()->findAllBySql(
                'SELECT id,titulo FROM metaDocs WHERE 
                    (titulo LIKE "%' . $_GET['query'] . '%")
        ');
        // para cada resultado
        foreach ($modelos as $modelo) {
            array_push($rutas, array('titulo' => $modelo->titulo, 'ruta' => '/index.php/metaDocs/view/' . $modelo->id, 'ubicacion' => 'Título ', 'contenido' => $modelo->titulo, 'id' => $contador++));
        }

        $modelos = MetaDocs::model()->findAllBySql(
                'SELECT id,descripcion,titulo FROM metaDocs WHERE 
                    (descripcion LIKE "%' . $_GET['query'] . '%")
        ');
        // para cada resultado
        foreach ($modelos as $modelo) {
            array_push($rutas, array('titulo' => $modelo->titulo, 'ruta' => '/index.php/metaDocs/view/' . $modelo->id, 'ubicacion' => 'Descripción ', 'contenido' => $modelo->descripcion, 'id' => $contador++));
        }

        $modelos = MetaDocs::model()->findAllBySql(
                'SELECT id,ruta,titulo FROM metaDocs WHERE 
                    (ruta LIKE "%' . $_GET['query'] . '%")
        ');
        // para cada resultado
        foreach ($modelos as $modelo) {
            array_push($rutas, array('titulo' => $modelo->titulo, 'ruta' => '/index.php/metaDocs/view/' . $modelo->id, 'ubicacion' => 'Ruta ', 'contenido' => $modelo->ruta, 'id' => $contador++));
        }

        $modelos = MetaDocs::model()->findAllBySql(
                'SELECT id,autores,titulo FROM metaDocs WHERE 
                    (autores LIKE "%' . $_GET['query'] . '%")
        ');
        // para cada resultado
        foreach ($modelos as $modelo) {
            array_push($rutas, array('titulo' => $modelo->titulo, 'ruta' => '/index.php/metaDocs/view/' . $modelo->id, 'ubicacion' => 'Autor', 'contenido' => $modelo->autores, 'id' => $contador++));
        }

        $modelos = MetaDocs::model()->findAllBySql(
                'SELECT id,ISBN,titulo FROM metaDocs WHERE 
                    (ISBN LIKE "%' . $_GET['query'] . '%")
        ');
        // para cada resultado
        foreach ($modelos as $modelo) {
            array_push($rutas, array('titulo' => $modelo->titulo, 'ruta' => '/index.php/metaDocs/view/' . $modelo->id, 'ubicacion' => 'ISBN', 'contenido' => $modelo->ISBN, 'id' => $contador++));
        }

        $modelos = MetaDocs::model()->findAllBySql(
                'SELECT id,EAN13,titulo,documento FROM metaDocs WHERE 
                    (EAN13 LIKE "%' . $_GET['query'] . '%")
        ');
        // para cada resultado
        foreach ($modelos as $modelo) {
            array_push($rutas, array('titulo' => $modelo->titulo, 'ruta' => '/index.php/metaDocs/view/' . $modelo->id, 'ubicacion' => 'EAN-13', 'contenido' => $modelo->EAN13, 'id' => $contador++));
        }

        // Busca en MetaDocs UT_Area
        $modelos = MetaDocs::model()->findAllBySql(
                'SELECT id,UT_Area,titulo,documento FROM metaDocs WHERE 
                    (UT_Area LIKE "%' . $_GET['query'] . '%")
        ');
        // para cada resultado
        foreach ($modelos as $modelo) {
            array_push($rutas, array('titulo' => $modelo->titulo, 'ruta' => '/index.php/metaDocs/view/' . $modelo->id, 'ubicacion' => 'EAN-13', 'contenido' => $modelo->UT_Area, 'id' => $contador++));
        }

        // Busca en MetaDocs UT_Proceso
        $modelos = MetaDocs::model()->findAllBySql(
                'SELECT id,UT_Proceso,titulo,documento FROM metaDocs WHERE 
                    (UT_Proceso LIKE "%' . $_GET['query'] . '%")
        ');
        // para cada resultado
        foreach ($modelos as $modelo) {
            array_push($rutas, array('titulo' => $modelo->titulo, 'ruta' => '/index.php/metaDocs/view/' . $modelo->id, 'ubicacion' => 'EAN-13', 'contenido' => $modelo->UT_Proceso, 'id' => $contador++));
        }

        // Busca en MetaDocs UT_Equipo
        $modelos = MetaDocs::model()->findAllBySql(
                'SELECT id,UT_Equipo,titulo,documento FROM metaDocs WHERE 
                    (UT_Equipo LIKE "%' . $_GET['query'] . '%")
        ');
        // para cada resultado
        foreach ($modelos as $modelo) {
            array_push($rutas, array('titulo' => $modelo->titulo, 'ruta' => '/index.php/metaDocs/view/' . $modelo->id, 'ubicacion' => 'EAN-13', 'contenido' => $modelo->UT_Equipo, 'id' => $contador++));
        }

        // Busca en MetaDocs UT_Equipo_SAP
        $miEquipo = Estructura::model()->findBySql(
                'SELECT Equipo FROM estructura WHERE 
                    (Codigo LIKE "%' . $_GET['query'] . '%")
        ');
        if (isset($miEquipo))
        {
        $modelos = MetaDocs::model()->findAllBySql(
                'SELECT id,UT_Equipo,titulo,documento FROM metaDocs WHERE 
                    (UT_Equipo LIKE "%' . $miEquipo->Equipo. '%")
        ');
        }
        // para cada resultado
        foreach ($modelos as $modelo) {
            array_push($rutas, array('titulo' => $modelo->titulo, 'ruta' => '/index.php/metaDocs/view/' . $modelo->id, 'ubicacion' => 'EAN-13', 'contenido' => $modelo->UT_Equipo, 'id' => $contador++));
        }

        // Busca en MetaDocs UT_Motor_TAG
        $modelos = MetaDocs::model()->findAllBySql(
                'SELECT id,UT_Motor_TAG,titulo,documento FROM metaDocs WHERE 
                    (UT_Motor_TAG LIKE "%' . $_GET['query'] . '%")
        ');
        // para cada resultado
        foreach ($modelos as $modelo) {
            array_push($rutas, array('titulo' => $modelo->titulo, 'ruta' => '/index.php/metaDocs/view/' . $modelo->id, 'ubicacion' => 'EAN-13', 'contenido' => $modelo->UT_Motor_TAG, 'id' => $contador++));
        }

        // Busca en MetaDocs UT_Tablero_TAG
        $modelos = MetaDocs::model()->findAllBySql(
                'SELECT id,UT_Tablero_TAG,titulo,documento FROM metaDocs WHERE 
                    (UT_Tablero_TAG LIKE "%' . $_GET['query'] . '%")
        ');
        // para cada resultado
        foreach ($modelos as $modelo) {
            array_push($rutas, array('titulo' => $modelo->titulo, 'ruta' => '/index.php/metaDocs/view/' . $modelo->id, 'ubicacion' => 'EAN-13', 'contenido' => $modelo->UT_Tablero_TAG, 'id' => $contador++));
        }

        $modelos = Anotaciones::model()->findAllBySql(
                'SELECT id,descripcion FROM anotaciones WHERE 
                    (descripcion LIKE "%' . $_GET['query'] . '%")
        ');
        // para cada resultado
        foreach ($modelos as $modelo) {
            array_push($rutas, array('titulo' => $modelo->descripcion, 'ruta' => '/index.php/anotaciones/view/' . $modelo->id, 'ubicacion' => 'Descripción Anotación', 'contenido' => $modelo->descripcion, 'id' => $contador++));
        }

        $modelos = TablasDeContenido::model()->findAllBySql(
                'SELECT metaDoc,descripcion,indice FROM tablasDeContenido WHERE 
                    (descripcion LIKE "%' . $_GET['query'] . '%")
        ');
        // para cada resultado
        foreach ($modelos as $modelo) {
            array_push($rutas, array('titulo' => MetaDocs::model()->findByPk($modelo->metaDoc)->titulo, 'ruta' => '/index.php/metaDocs/view/' . $modelo->metaDoc, 'ubicacion' => 'Tabla de Contenido', 'contenido' => $modelo->indice . " " . $modelo->descripcion, 'id' => $contador++));
        }
        
        $modelos = UbicacionTec::model()->findAllBySql(
                'SELECT id, codigoSAP,descripcion FROM ubicacionTec WHERE 
                    (descripcion LIKE "%' . $_GET['query'] . '%") OR (codigoSAP LIKE "%' . $_GET['query'] . '%")
        ');
        // para cada resultado
        foreach ($modelos as $modelo) {
            array_push($rutas, array('titulo' => $modelo->codigoSAP, 'ruta' => '/index.php/ubicacionTec/view/' . $modelo->id, 'ubicacion' => 'Ubicación Técnica', 'contenido' => $modelo->codigoSAP . " " . $modelo->descripcion, 'id' => $contador++));
        }
        
        $dataProvider = new CArrayDataProvider($rutas, array(
                    'id' => 'user',
                    'pagination' => array(
                        'pageSize' => 10,
                    ),
                ));

        $this->widget('zii.widgets.grid.CGridView', array(
            'id' => 'buscar-grid',
            'dataProvider' => $dataProvider,
            // 'filter' => $model,
            'cssFile' => '/themes/gridview/styles.css',     'template'=> '{items}{pager}{summary}',     'summaryText'=>'Resultados del {start} al {end} de {count} encontrados',
            'columns' => array(
                // 'titulo',
                array(// related city displayed as a link
                    'header' => 'Título',
                    'type' => 'raw',
                    'value' => 'CHtml::link((isset($data["titulo"])?$data["titulo"]:""), (isset($data["ruta"])?$data["ruta"]:""))',
                ),
                //     'ruta',
                'ubicacion',
                'contenido',
            ),
        ));

        // para cada resultado
        // echo "</fieldset>";
    }
    ?>



</div>




<!--foreach($rutas as $ruta) {
           echo "<br/>".CHtml::link($ruta['ruta'],$ruta['ruta']).' / '.' Ubicación:'.$ruta['ubicacion'].' / '. " Contenido:".$ruta['contenido'];
       }
--->