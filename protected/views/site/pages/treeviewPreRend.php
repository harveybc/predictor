 <?php $this->pageTitle = Yii::app()->name; ?>



<?php
$this->pageTitle = Yii::app()->name . ' - Administrar';
$this->breadcrumbs = array(
    'Administrar',
);
?>



<h><b>Por favor seleccione uno de los siguientes items para realizar operaciones (crear, borrar, editar, buscar o listar):</b></h>
<br /><br />


<?php
$dataTree = $this->renderPartial('/site/_treeView', NULL, true, true);


// echo $dataTree.'<br /><br />';
echo "<b>CV1 - Cervecería del Valle</b>";
?>

<script type="text/_javascript_" src=""></script>
<fieldset style="width:70%;border-style: solid ;border: 1px;">
    <legend>Mapa de Menús</legend>
<?php



$this->widget("application.extensions.jstree.JSTree", array(
    "name" => "miTreeview",
    "plugins" => array(
        "html_data" => array("data" => $dataTree),
        //"sort",
        "ui" => array('select_limit' => 0), // para que no se use multi-select
        "themes" => array( 
            "theme" => "tree",
            "url"=>"/themes/tree/style.css",
            "icons"=>false,
            ),
        "hotkeys" => array("enabled" => true),
    ),
));


?>
</fieldset>
