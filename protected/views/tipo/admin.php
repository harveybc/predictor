
<?php
//para leer el param get y con el reconfigurar los dropdown
$sufix = "";
$valor = "";
if (isset($_GET['area'])) {
    $valor = $_GET['area'];
    $sufix = "?id=" . $_GET['area'];
}
?>

<style type="text/css">



    .select_admin{

        background:#ffffff;        
        border: 1px solid #DBC08F;
        -moz-border-radius:3px;
        -webkit-border-radius: 2px;
        border-radius:2px;


    }



</style> 

<?php
$this->breadcrumbs = array(
    'Tipos' => array(Yii::t('app', 'index')),
    Yii::t('app', 'Gestionar'),
);

$this->menu = array(
    array('label' => Yii::t('app', 'Lista de Lubricantes'), 'url' => array('index')),
    array('label' => Yii::t('app', 'Nuevo Lubricante'),
        'url' => array('create' . $sufix),
        'itemOptions' => array('id' => 'linkNuevo')),
);

Yii::app()->clientScript->registerScript('search', "
			$('.search-button').click(function(){
				$('.search-form').toggle();
				return false;
				});
			$('.search-form form').submit(function(){
				$.fn.yiiGridView.update('tipo-grid', {
data: $(this).serialize()
});
				return false;
				});
			");
?>

<?php $this->setPageTitle (' Gestionar&nbsp;Lubricantes'); ?>
<div class="forms50c">

            <b>Area:</b>
            <?php
// dibuja el dropDownList de Proceso, seleccionando los valores diferentes presentes en la tabla Estructura col. Proceso

            /*
              echo CHtml::activeDropDownList(
              $model,'Lubricante', CHtml::listData(Estructura::model()->findAllbySql(
              'SELECT * Proceso FROM estructura ORDER BY Proceso ASC', array()), 'Proceso', 'Proceso'
              ),

             * * 
             * 
             */
            echo CHtml::dropDownList(
                    'proceso', $valor, CHtml::listData(Estructura::model()->findAllbySql(
                                    'SELECT DISTINCT Proceso FROM estructura ORDER BY Proceso ASC', array()), 'Proceso', 'Proceso'
                    ), array(
                'onchange' => 'updateLink()',
                'ajax' => array(
                    'type' => 'GET', //request type
                    'data' => array('proceso' => 'js:document.getElementById("proceso").value'),
                    'url' => CController::createUrl('/tipo/dynamicFechas'), //url to call.
                    'update' => '#divFechas', //selector to update
                ),
                'style' => 'width:80%;',
                'class' => 'select',
                '' => 'select',
                'empty' => 'Seleccione el Area'
                    )
            );
            ?>


<?php echo CHtml::link(Yii::t('app', 'Búsqueda Avanzada'), '#', array('class' => 'search-button','style'=>'display:block;')); ?>
<div class="search-form" style="display:none">
    <?php
    $this->renderPartial('_search', array(
        'model' => $model,
    ));
    ?>
</div>
    </div>
            <script type="text/javascript">
    // función que actualiza link de crear
    function updateLink()
    {
        $('#linkNuevo').html('<a href="/index.php/tipo/create?id='+document.getElementById("proceso").value+'">Nuevo Lubricante</a>');
    }
    // función que actualiza el campo de Area dependiendo del campo de proceso
    function updateGrid()
    {
                
<?php
echo CHtml::ajax(array(
    'type' => 'GET', //request type
    'data' => array('proceso' => 'js:document.getElementById("proceso").value'),
    'url' => CController::createUrl('/tipo/dynamicFechas'), //url to call.
    'update' => '#divFechas', //selector to update
        )
);
?>
        //document.getElementById('Examenes_convenio').selectedIndex = conv;

        return false;
    }
</script>
<div id="divFechas" name="divFechas">

    <?php
    if (isset($_GET['area'])) {
        echo "<script language=javascript>updateGrid()</script>";
    } else {

        /*
          $this->widget('zii.widgets.grid.CGridView', array(
          'id' => 'clinicas-grid',
          'dataProvider' => $model->search(),
          //  'filter' => $model,
          'cssFile' => '/themes/gridview/styles.css',     'template'=> '{items}{pager}{summary}',     'summaryText'=>'Resultados del {start} al {end} de {count} encontrados',
          'columns' => array(
          'Tipo_Aceite',
          'Proceso',
          array(
          'class' => 'CButtonColumn',
          ),
          ),
          ));
         */
    }
    ?>

</div>