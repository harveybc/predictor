<?php $this->pageTitle = Yii::app()->name; ?>



<?php
    $this->pageTitle = Yii::app()->name . ' - Administrar';
    $this->breadcrumbs = array (
        'Administrar',
    );
?>



<h><b>Bienvenido a CBM</b></h>
<br /><br />


<?php
    $boton = CHtml::linkButton('Ver',
                               array ('submit' => '/index.php/motores/admin'));
    $treeData =
        '<li>
	 <a href="some_value_here">ELABORACIÓN</a>SEPARATOR' . $boton . '
	<!-- UL node only needed for children - omit if there are no children -->
	<ul>
		<!-- Children LI nodes here -->
                <li>
                        Nodo 2<a href="some_value_here">Node title</a>
                        <!-- UL node only needed for children - omit if there are no children -->
                </li>
	</ul>
</li>
<li>
	<a href="some_value_here">ENVASE </a>
	<!-- UL node only needed for children - omit if there are no children -->
	<ul>
		<!-- Children LI nodes here -->
                <li>
                        Nodo 2<a href="some_value_here">Node title</a>
                        <!-- UL node only needed for children - omit if there are no children -->
                </li>
	</ul>
</li>
<li>
	<a href="some_value_here">SERVICIOS </a>
	<!-- UL node only needed for children - omit if there are no children -->
	<ul>
		<!-- Children LI nodes here -->
                <li>
                        Nodo 2<a href="some_value_here">Node title</a>
                        <!-- UL node only needed for children - omit if there are no children -->
                </li>
	</ul>
</li>
';

// CampoArea() Esta función Imprime el enlace y las opciones para cada Área, el
// Parámetros:
// cArea = Nombre del area


    function CampoArea($cArea)
    {
        // inicializa cadena vacía
        $cadSalida = "";
        // Adiciona Título 
        $cadSalida = $cadSalida . CHtml::link($cArea->Proceso
                . CHtml::linkButton('Detalles',
                                    array (
                    'submit' => '/index.php/site/page?view=resumen&area=' . $cArea->Proceso,
                    'style' => 'background-color:#E8D985;opacity: 0.70 ;background-image:none !important;border-left:2px solid #ffffff;border-right:2px solid #ffffff;width:85px;text-align:center;'
                ))
                . CHtml::linkButton('Lubricantes',
                                    array (
                    'submit' => '/index.php/tipo/admin?area=' . $cArea->Proceso,
                    'style' => 'background-color:#E8D985;opacity: 0.70 ;background-image:none !important;width:84px;text-align:center;border-left:2px solid #ffffff;border-right:2px solid #ffffff;'
                ))
                ,
                                    "#",
                                    array ('style' => 'width:663px;background-color:#transparent;background-image:none !important;'));
        // Adiciona línea horizontal de abajo
        //$cadSalida=$cadSalida.$boton = CHtml::linkButton('Ver', array('submit' => '/index.php/motores/admin'));
        return($cadSalida);
    }

// CampoProceso() Esta función Imprime el enlace y las opciones para cada Proceso, el
// Parámetros:
// cProceso = Nombre del proceso
    function CampoProceso($cProceso)
    {
        $cadSalida = "";
        $cadSalida = $cadSalida . CHtml::link($cProceso->Area,
                                              "#",
                                              array ('style' => 'width:645px;background-color:#transparent;background-image:none !important'))
            . CHtml::linkButton('Estr. Tableros',
                                array (
                'submit' => '/index.php/tableros/admin?proceso=' . $cProceso->Area,
                'style' => 'background-color:#D6DCE8;opacity: 0.70 ;background-image:none !important;width:84px;text-align:center;border-left:2px solid #ffffff;border-right:2px solid #ffffff;'
            ))
            . CHtml::linkButton('Equipos',
                                array (
                'submit' => '/index.php/estructura/admin?proceso=' . $cProceso->Area,
                'style' => 'background-color:#D6DCE8;opacity: 0.70 ;background-image:none !important;width:85px;text-align:center;border-left:2px solid #ffffff;border-right:2px solid #ffffff;'
            ))
        ;
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
    foreach ($areas as $area)
    {
        $dataTree = $dataTree . "<li>"; // Area
        $dataTree = $dataTree . CampoArea($area);
        $dataTree = $dataTree . "<ul>"; // Area
        $procesos = Estructura::model()->findAllBySql('SELECT DISTINCT Area FROM estructura WHERE Proceso="' . $area->Proceso . '" ORDER BY Area');
        // para cada proceso
        $np = 0;
        foreach ($procesos as $proceso)
        {
            $dataTree = $dataTree . "<li>"; // Proceso
            $dataTree = $dataTree . CampoProceso($proceso);
            $dataTree = $dataTree . "<ul>"; // Proceso
            //$dataTree = $dataTree . '<li id="' . $proceso->Proceso. '" class="jstree-closed">'; // Letrero de Tablero
            $dataTree = $dataTree . '<li id="'.$proceso->Area.'" tipo="Tablero" class="jstree-closed">'; // Letrero de Tablero
            $dataTree = $dataTree . '<a href="#" style="width: 97%;">Tableros</a>';
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
            $dataTree = $dataTree . '<li id="' . $proceso->Area . '" tipo="Equipos" class="jstree-closed">'; // Letrero de Equipos
            $dataTree = $dataTree . '<a href="#" style="width: 97%;">Equipos</a>';
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
    echo "<b>CV1 - Cervecería del Valle</b>";
?>
<form>
    <input type="hidden" id="proceso" value="OK"/>
</form>
<script type="text/_javascript_" src=""></script>
<fieldset style="width:870px;border-style: solid ;border: 1px;">
<?php
    $this->widget("application.extensions.jstree.JSTree",
                  array (
        "name" => "miTreeview",
        "plugins" => array (
            "html_data" => array (
                "data" => $dataTree,
                "ajax" => array (
                    "data" => 'js:function(n)
                    {
                        return{"id" : n.attr ? n.attr("id") : 1, "tipo" : n.attr ? n.attr("tipo") : 1};
                    }
                ',
                    "url" => CController::createUrl('/site/getArbol'),
                )
            ),
            //"sort",
            "ui" => array ('select_limit' => 0), // para que no se use multi-select
            "themes" => array (
                "theme" => "tree",
                "url" => "/themes/tree/style.css",
                "icons" => false,
            ),
            "hotkeys" => array ("enabled" => true),
        ),
    ));
?>
</fieldset>
<?php
    /*
      $dataTree=array(
      array(
      'text'=>'Grampa', //must using 'text' key to show the text
      'children'=>array(//using 'children' key to indicate there are children
      array(
      'text'=>'Father',
      'children'=>array(
      array('text'=>'<hr />maaIe<hr />'),
      array('text'=>'big sis<hr />'),
      array('text'=>'little brother<hr />'),
      )
      ),
      array(
      'text'=>'Uncle',
      'children'=>array(
      array('text'=>'Ben'),
      array('text'=>'Sally'),
      )
      ),
      array(
      'text'=>'Aunt',
      )
      )
      )
      );

     * 
     * 
     */
    /*
      // coloca el datatree vacío.
      $dataTree=array();
      // encuentra todas las áreas (columna proceso)
      $areas=Estructura::model()->findAllBySql("SELECT DISTINCT Proceso FROM estructura ORDER BY Proceso");
      // para cada área adiciona una entrada y sus acciones,
      $na=0;
      foreach($areas as $area){
      array_push($dataTree, array(
      // faltan opciones
      'data'=>$area->Proceso,
      'children'=>array(),
      ));
      $procesos=Estructura::model()->findAllBySql('SELECT DISTINCT Area FROM estructura WHERE Proceso="'.$area->Proceso.'" ORDER BY Area');
      // para cada proceso
      $np=0;
      foreach ($procesos as $proceso){
      array_push($dataTree[$na]['children'],array(
      'data'=>$proceso->Area,
      'children'=>array(
      array('data'=>'Equipos','children'=>array()),
      array('data'=>'Tableros','children'=>array()),
      ),
      ));
      $tableros=  Tableros::model()->findAllBySql('SELECT Tablero FROM tableros WHERE Area="'.$proceso->Area.'" ORDER BY Tablero');
      // para cada tablero
      foreach ($tableros as $tablero){
      array_push($dataTree[$na]['children'][$np]['children'][1]['children'],array(
      'data'=>$tablero->Tablero,
      'children'=>array(),
      ));
      }
      $equipos=Estructura::model()->findAllBySql('SELECT Equipo FROM estructura WHERE Area="'.$proceso->Area.'" ORDER BY Equipo');
      // para cada proceso
      $ne=0;
      foreach ($equipos as $equipo){
      array_push($dataTree[$na]['children'][$np]['children'][0]['children'],array(
      'data'=>$equipo->Equipo,
      'children'=>array(),
      ));
      $motores=Motores::model()->findAllBySql('SELECT Motor FROM motores WHERE Equipo="'.$equipo->Equipo.'" ORDER BY Motor');
      foreach ($motores as $motor){
      array_push($dataTree[$na]['children'][$np]['children'][0]['children'][$ne]['children'],array(
      'data'=>$motor->Motor.' - '.$motor->TAG,
      'children'=>array(),
      ));
      }
      $ne++;
      }
      $np++;
      }
      $na++;
      }
      $jsonArr=json_encode($dataTree);
      //print_r($dataTree);
     * 
     */


    /*
      // agrega el arreglo de hijos
      $procesos=Estructura::model()->findAllBySql('SELECT DISCTINT Area FROM estructura WHERE Proceso="'.$area->Proceso.'" ORDER BY Area');
      // para cada área adiciona una entrada y sus acciones,

      foreach($procesos as $proceso){
      array_push($dataTree, array(
      $dataTree['']
      ));

      // para los equipos del proceso, adiciona una entrada y sus acciones,
      // para los motores del equipo, adiciona una entrada y sus acciones
      }


      $this->widget('CTreeView', array(
      'htmlOptions' => array(
      'class' => 'treeview-famfamfam'
      ),
      'id' => 'treeviewId',

      'data' =>$dataTree,



      'unique' => true,
      //'persist' => 'cookie',
      //'prerendered' => false,}
      'collapsed' => true,
      //'control' => '#selector',
      //'url' => array('ajaxFillTree')
      ));
     */

    /*
      $jsonTest=json_encode(array('data'=>'TEST'));
      $this->widget("application.extensions.jstree.JSTree", array(
      'name'=>'jsonTree',
      "plugins"=>array(
      "html_data"=>'<li>
      <a href="some_value_here">Node title</a>

      </li>
      ',
      "themes"=>array( "theme" => "default" ),
      "sort",
      "ui"
      ),
      ));
      echo "end";

     * 
     */
?>



