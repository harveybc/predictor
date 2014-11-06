<?php
    $sufix="";
    if (isset($_GET['id']))
    {
        $modeloId = Estructura::model()->findByAttributes(array ("id" => $_GET['id']));
        if (isset($modeloId))
            $sufix = "?id=" . urlencode($modeloId->Equipo);
    }
    /*
      $this->widget( 'ext.EChosen.EChosen', array(
      'target' => 'select',
      'useJQuery' => true,
      'debug' => false,
      ));
     */
?>

<style type="text/css">

div.loading {
    background-color: #FFFFFF;
    background-image: url('/images/loading.gif');
    background-position:  100px;
    background-repeat: no-repeat;
    opacity: 1;
}
div.loading * {
    opacity: .8;
}

    .select_admin{

        width: 30px;
        padding-left:2px;
        background:#ffffff;
        background-color:#ffffff;
        border: 1px solid #DBC08F;
        -moz-border-radius:3px;
        -webkit-border-radius: 3px;
        border-radius:3px;


    }
    .back{

               
       padding-left:605px


    }




</style>



<script type="text/javascript">
    // función que actualiza el campo de Area dependiendo del campo de proceso
    function updateFieldArea()
    {
<?php
    echo CHtml::ajax(array (
        'type' => 'GET', //request type
        'data' => array ('proceso' => 'js:document.getElementById("proceso").value'),
        'url' => CController::createUrl('/motores/dynamicArea'), //url to call.
        //'update' => '#Visitas_idDoctor', //selector to update
        'success' => 'updateAreaDropdown',
        'beforeSend' => 'function(){
      $("#myDiv").addClass("loading");}',
    'complete' => 'function(){
      $("#myDiv").removeClass("loading");}',
        'dataType' => 'json')
    );
?>
        //document.getElementById('Examenes_convenio').selectedIndex = conv;
        return false;
    }

    function updateFieldEquipo()
    {
<?php
    echo CHtml::ajax(array (
        'type' => 'GET', //request type
        'data' => array ('area' => 'js:document.getElementById("area").value'),
        'url' => CController::createUrl('/motores/dynamicEquipo'), //url to call.
        //'update' => '#Visitas_idDoctor', //selector to update
        'success' => 'updateEquipoDropdown',
        'beforeSend' => 'function(){
      $("#myDiv").addClass("loading");}',
    'complete' => 'function(){
      $("#myDiv").removeClass("loading");}',
        'dataType' => 'json')
    );
?>
        //document.getElementById('Examenes_convenio').selectedIndex = conv;
        return false;
    }

    function updateGridMotores()
    {
                $('#linkNuevo').html('<a href="/index.php/motores/create?id='+encodeURIComponent(document.getElementById("equipo").value)+'">Nuevo Motor</a>');
<?php
    echo CHtml::ajax(array (
        'type' => 'GET', //request type
        'data' => array ('area' => 'js:document.getElementById("area").value', 'equipo' => 'js:document.getElementById("equipo").value'),
        'update' => '#gridMotores',
        'url' => CController::createUrl('/motores/dynamicMotores'), //url to call.
        //'update' => '#Visitas_idDoctor', //selector to update
        //'success' => 'updateContGridMotores',
        //'dataType' => 'json'
        )
    );
?>
        //document.getElementById('Examenes_convenio').selectedIndex = conv;
        return false;
    }

    // función que cambia los datos de el Dropdownlist de Area
    function updateAreaDropdown(data)
    {
        $('#area').html(data.value1);
        // actualiza el echosen de area

        updateFieldEquipo();
    };

    // función que cambia los datos de el Dropdownlist de Equipo
    function updateEquipoDropdown(data)
    {
        $('#equipo').html(data.value1);

        //updateGridMotores();
    };


</script>
<?php
    $this->breadcrumbs = array (
        'Motores' => array (Yii::t('app',
                                   'index')),
        Yii::t('app',
               'Gestionar'),
    );

    $this->menu = array (
        array ('label' => Yii::t('app','Lista de Motores'), 'url' => array ('index')),
        
      );
//TODO: provisional: para uso de roles de admin, ingeniero y usuario.
$esAdmin = 0;
$esIngeniero = 0;
if (!Yii::app()->user->isGuest) {
    $modeloU = Usuarios::model()->findBySql('select * from usuarios where Username="' . Yii::app()->user->name . '"');
}
if (isset($modeloU)) {
    $esAdmin = $modeloU->Es_administrador;
    $esIngeniero = $modeloU->Es_analista;
    if ($esAdmin)
        $esIngeniero = 1;
}
if ($esIngeniero)
    array_push($this->menu, array ('label' => Yii::t('app','Nuevo Motor'), 'url' => array ('create'. $sufix),
            'itemOptions' => array('id' => 'linkNuevo')));
    
    Yii::app()->clientScript->registerScript('search',
                                             "
			$('.search-button').click(function(){
				$('.search-form').toggle();
				return false;
				});
			$('.search-form form').submit(function(){
				$.fn.yiiGridView.update('motores-grid', {data: $(this).serialize()});
				return false;
				});
			");
?>
<?php $this->setPageTitle ('Gestionar&nbsp;Motores'); ?>

<div name="myDiv" id="myDiv" class="forms100cb">
<form>
    <table style="width:100%; ">
        <tr>
            <td style="width:50%;">
                <b>Area:</b>
                <?php
                    $valor = isset($modeloId) ? $modeloId->Proceso : "";
// dibuja el dropDownList de Proceso, seleccionando los valores diferentes presentes en la tabla Estructura col. Proceso
                    echo CHtml::dropDownList(
                        'proceso',
                        $valor,
                        CHtml::listData(Estructura::model()->findAllbySql(
                                'SELECT DISTINCT Proceso FROM estructura ORDER BY Proceso ASC',
                                array ()),
                                'Proceso',
                                'Proceso'
                        ),
                                array (
                        //'onfocus' => 'updateFieldArea()',
                        'onchange' => 'updateFieldArea()',
                        'style' => 'width:100%;',
                        'class' => 'select'
                        )
                    );
                ?>
                <!-- an la app original era:SELECT DISTINCT Area , Indicativo FROM Estructura WHERE (Proceso=@Proceso) ORDER BY Indicativo ASC -->
            </td>
            <td  style="width:50%;">
                <b>Proceso:</b>
                <?php
                    $valor = isset($modeloId) ? $modeloId->Area : "";
                    // dibuja el dropDownList de Area, dependiendo del proceso selecccionado
                    echo CHtml::dropDownList(
                        'area',
                        $valor,
                        CHtml::listData(Estructura::model()->findAllbySql(
                                isset($modeloId) ?
                                    'SELECT DISTINCT Area FROM estructura WHERE Proceso="' . $modeloId->Proceso . '" ORDER BY Area ASC'
                                        : 'SELECT DISTINCT Area FROM estructura WHERE Proceso="ELABORACION" ORDER BY Area ASC'
                                ,
                                      array ()),
                                      'Area',
                                      'Area'
                        ),
                                      array (
                        //'onfocus' => 'updateFieldArea()',
                        'onchange' => 'updateFieldEquipo()',
                        'style' => 'width:100%;',
                        'class' => 'select',
                        'empty' => 'Seleccione el proceso',
                        )
                    );
                ?>
            </td>
        </tr>
    </table>


    <table style="width:100%; ">
        <tr>
            <td>
                <b>Equipo:<br/></b>
                <?php
                    $valor = isset($modeloId) ? $modeloId->Equipo : "";
                    // dibuja el dropDownList de Proceso, seleccionando los valores diferentes presentes en la tabla Estructura col. Proceso
                    echo CHtml::dropDownList(
                        'equipo',
                        $valor,
                        CHtml::listData(Estructura::model()->findAllbySql(
                                isset($modeloId) ?
                                    'SELECT Equipo FROM estructura WHERE Area="' . $modeloId->Area . '" ORDER BY Equipo ASC'
                                        : 'SELECT Equipo FROM estructura WHERE Area="FILTRACION" ORDER BY Equipo ASC',
                                      array ()),
                                      'Equipo',
                                      'Equipo'
                        ),
                                      array (
                        //'onfocus' => 'updateFieldEquipo()',
                        'onchange' => 'updateGridMotores()',
                        'style' => 'width:100%;',
                        'class' => 'select',
                        'empty' => 'Seleccione el equipo para filtrar el resultado',
                        )
                    );
                ?>
            </td>
        </tr>
    </table>
</form>
    
    </div>
        
<?php
    echo CHtml::link(Yii::t('app',
                            'Búsqueda Avanzada'),
                            '#',
                            array ('class' => 'search-button'));
?>
<div class="search-form" style="display:none">
    <?php
        $this->renderPartial('_search',
                             array (
            'model' => $model,
            )
        );
    ?>
</div>

<div id="gridMotores" name="gridMotores">
    <?php
        if (isset($modeloId))
        {
            echo "<script language=javascript>updateGridMotores()</script>";
        }
        else
        {
 /*
            $this->widget('zii.widgets.grid.CGridView',
                          array (
                'id' => 'clinicas-grid',
                'dataProvider' => $model->search(),
                //  'filter' => $model,
                'cssFile' => '/themes/gridview/styles.css',     'template'=> '{items}{pager}{summary}',     'summaryText'=>'Resultados del {start} al {end} de {count} encontrados',
                'columns' => array (
                    'TAG',
                    'Equipo',
                    'Motor',
                    'Marca',
                    'Modelo',
                    'Serie',
                    'Lubricante',
                    array (
                        'class' => 'CButtonColumn',
                    ),
                ),
            ));
  */      }
    ?>
</div>